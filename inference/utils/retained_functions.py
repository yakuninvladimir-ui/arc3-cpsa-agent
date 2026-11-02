"""Conservative source-only function retention; no snippet state is serialized.

Also embedded into the isolated Python runner. Keep this module stdlib-only.
"""
import ast
import dis
import sys
import types


def _is_immutable_literal(val):
    if isinstance(val, (int, float, str, bytes, bool, type(None))):
        return True
    if isinstance(val, (tuple, frozenset)):
        return all(_is_immutable_literal(x) for x in val)
    return False


def _import_statements(nodes):
    """One explicit absolute import per binding; never replay wildcard imports."""
    statements = {}
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                binding = alias.asname or alias.name.split('.')[0]
                statements[binding] = ast.unparse(ast.Import(names=[alias]))
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            for alias in node.names:
                if alias.name == '*':
                    raise ValueError('wildcard imports are not retained; use explicit imports inside the function')
                statements[alias.asname or alias.name] = ast.unparse(
                    ast.ImportFrom(module=node.module, names=[alias], level=0))
        else:
            raise ValueError('only explicit absolute imports can accompany retained functions')
    return statements


def _split_preamble(nodes):
    import_nodes = []
    const_bindings = {}
    for node in nodes:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            import_nodes.append(node)
        elif (isinstance(node, ast.Assign) and len(node.targets) == 1
              and isinstance(node.targets[0], ast.Name)):
            try:
                val = ast.literal_eval(node.value)
            except Exception:
                raise ValueError('only literal constants and explicit imports can accompany retained functions') from None
            if not _is_immutable_literal(val):
                raise ValueError('only immutable literal constants can accompany retained functions')
            name = node.targets[0].id
            const_bindings[name] = f"{name} = {repr(val)}"
        else:
            raise ValueError('only explicit absolute imports can accompany retained functions')
    imports = _import_statements(import_nodes)
    return imports, const_bindings


def _source_parts(source, retain_imports=True):
    tree = ast.parse(source, '<retained>', 'exec')
    if (not tree.body or not isinstance(tree.body[-1], ast.FunctionDef)
            or (not retain_imports and len(tree.body) != 1)):
        raise ValueError('only top-level function definitions can be retained')
    imports, consts = _split_preamble(tree.body[:-1])
    combined = {**imports, **consts}
    return tree.body[-1], combined


def function_signature(source):
    node, _ = _source_parts(source)
    return node.name + '(' + ast.unparse(node.args) + ')'


def _function_record(source, retain_imports=False):
    node, imports = _source_parts(source, retain_imports)
    if getattr(node, "type_params", []):
        raise ValueError("generic type parameters are not retained")
    if node.decorator_list:
        raise ValueError('decorated functions are not retained')
    args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    args += [a for a in (node.args.vararg, node.args.kwarg) if a is not None]
    if node.returns is not None or any(a.annotation is not None for a in args):
        raise ValueError('use an unannotated definition for retention')
    # Restoration must never evaluate model calls or snapshot a frame via defaults.
    for default in [*node.args.defaults, *node.args.kw_defaults]:
        if default is not None:
            try:
                ast.literal_eval(default)
            except Exception:
                raise ValueError('defaults must be literals; pass changing data as arguments') from None
    tree = ast.Module(body=[node], type_ignores=[])
    code = next(c for c in compile(tree, '<retained>', 'exec').co_consts
                if isinstance(c, types.CodeType))
    reads = set()
    pending = [code]
    while pending:
        current = pending.pop()
        for instruction in dis.get_instructions(current):
            if instruction.opname in ('LOAD_GLOBAL', 'LOAD_NAME'):
                reads.add(instruction.argval)
            elif instruction.opname in ('STORE_GLOBAL', 'DELETE_GLOBAL'):
                raise ValueError('functions that assign global variables are not retained')
        pending.extend(c for c in current.co_consts if isinstance(c, types.CodeType))
    return node.name, source, reads - imports.keys()


def _rejection(name, reason, hint=None, **metadata):
    result = {'name': name, 'reason': reason, **metadata}
    if hint:
        result['hint'] = hint
    return result


def _validation_hint(reason):
    if 'decorat' in reason:
        return 'Use a plain top-level def without decorators if you want to reuse it.'
    if 'annotat' in reason:
        return 'Remove parameter and return annotations if you want to retain this function.'
    if 'default' in reason:
        return 'Use literal defaults and pass changing values explicitly as arguments.'
    if 'global variables' in reason:
        return 'Return updated values instead of modifying global state.'
    return None


def prepare_functions(sources, available, *, retain_imports=False):
    """Validate candidates together; return (definitions, rejected_records).

    Order is preserved: early functions can serve as helpers to later ones.
    A failure on one candidate must never discard the valid helpers it called.
    """
    valid = {}
    dependencies = {}
    rejected = []
    for source in sources:
        try:
            name, _, reads = _function_record(source, retain_imports=retain_imports)
        except Exception as exc:
            reason = str(exc) or exc.__class__.__name__
            try:
                tree = ast.parse(source)
                for node in reversed(tree.body):
                    if isinstance(node, ast.FunctionDef):
                        rejected.append(_rejection(node.name, reason, _validation_hint(reason)))
                        break
            except Exception:
                pass
            continue
        valid[name] = source
        dependencies[name] = reads
    # Drop candidates that read names neither passed in nor defined earlier.
    provided = set(available) | set(dir(sys.modules['builtins']))
    resolved = set(provided)
    changed = True
    while changed:
        changed = False
        for name, reads in list(dependencies.items()):
            if name not in resolved and reads.issubset(resolved):
                resolved.add(name)
                changed = True
    invalid = []
    for name in list(dependencies):
        if name not in resolved:
            missing = sorted(dependencies[name] - resolved)
            invalid.append(_rejection(name, f'unavailable dependencies: {", ".join(missing)}',
                                      missing_dependencies=missing))
            del valid[name]
    return list(valid.values()), rejected + invalid


def _matching_imports(nodes, namespace):
    """Imports declared in the snippet matching live module bindings."""
    matches = {}
    declared = {}
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split('.')[0]
                declared[name] = ast.unparse(ast.Import(names=[alias]))
                if name in namespace and isinstance(namespace[name], types.ModuleType):
                    matches[name] = declared[name]
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            for alias in node.names:
                if alias.name == '*':
                    continue
                name = alias.asname or alias.name
                declared[name] = ast.unparse(ast.ImportFrom(module=node.module, names=[alias], level=0))
                if name in namespace:
                    matches[name] = declared[name]
    return matches, declared


def collect_functions(code_text, previous, namespace, available, *, retain_imports=False,
                      repair_hints=False):
    """Collect from the actual namespace, including after partial execution.

    A later def may never have run. Match the live function's source location
    rather than choosing the last definition in the submitted snippet.
    """
    candidates = {}
    rejected = []
    try:
        tree = ast.parse(code_text, '<python_tool>')
    except SyntaxError:
        # Parsing failed before execution; restored functions are still valid.
        tree = ast.Module(body=[], type_ignores=[])
    fresh = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fresh.setdefault(node.name, []).append(node)
    import_nodes = []
    for source in previous.values():
        # Old source-only records remain valid; no migration is necessary.
        import_nodes.extend(ast.parse(source).body[:-1])
    import_nodes.extend(tree.body)
    imports, declared_imports = (_matching_imports(import_nodes, namespace)
                                if retain_imports or repair_hints else ({}, {}))
    for name in dict.fromkeys([*previous, *fresh]):
        value = namespace.get(name)
        if not repair_hints and (name in available or type(value) is not types.FunctionType
                                 or value.__name__ != name):
            rejected.append(_rejection(name, 'definition did not execute, was removed/replaced, or conflicts with a provided name'))
            continue
        if name in available:
            rejected.append(_rejection(name, 'name conflicts with a provided global or builtin',
                                       'Choose a different function name.'))
            continue
        if name not in namespace:
            reason = 'function was removed' if name in previous else 'definition was not reached or was removed'
            rejected.append(_rejection(name, reason, 'Define it before calling it if it is still needed.'))
            continue
        if type(value) is not types.FunctionType or value.__name__ != name:
            rejected.append(_rejection(name, 'function name was replaced by another value',
                                       'Use a separate variable name; redefine the function only if needed.'))
            continue
        if name in previous and value.__code__.co_filename == '<retained>':
            # Preserve the restored definition when execution stopped before a
            # replacement (or while evaluating that replacement's defaults).
            candidates[name] = previous[name]
            continue
        node = next((node for node in fresh.get(name, [])
                     if isinstance(node, ast.FunctionDef) and not node.decorator_list
                     and value.__code__.co_filename == '<python_tool>'
                     and value.__code__.co_firstlineno == node.lineno), None)
        if node is None:
            rejected.append(_rejection(name, 'definition was decorated, replaced outside a top-level def, or is unsupported',
                                       'Use a plain top-level def without decorators if you want to reuse it.'))
            continue
        candidates[name] = ast.get_source_segment(code_text, node)
    # Collect live immutable constants from namespace
    for k, v in list(namespace.items()):
        if (isinstance(k, str) and k.isidentifier() and k not in available
                and k not in imports and k not in candidates
                and _is_immutable_literal(v) and len(repr(v)) <= 400):
            imports[k] = f"{k} = {repr(v)}"
    # Rebuild every bundle from the LIVE namespace. An old helper must not
    # silently restore an obsolete import when that alias was rebound/deleted.
    for name, source in (list(candidates.items()) if retain_imports else []):
        try:
            node, _ = _source_parts(source)
            definition = ast.get_source_segment(source, node)
            _, _, reads = _function_record(definition)
            statements = [imports[n] for n in sorted(reads & imports.keys()) if n not in available]
            candidates[name] = '\n'.join([*statements, definition])
        except Exception:
            # Validation below produces the specific rejection and prunes callers.
            pass
    kept, invalid = prepare_functions(candidates.values(), available, retain_imports=retain_imports)
    for item in invalid:
        missing = item.get('missing_dependencies', [])
        if not missing:
            continue
        hints = []
        missing_imports = [n for n in missing if n in declared_imports]
        missing_helpers = [n for n in missing if n in candidates and n not in declared_imports]
        missing_data = sorted(set(missing) - set(missing_imports) - set(missing_helpers))
        if missing_imports:
            statements = '; '.join(declared_imports[n] for n in missing_imports)
            hints.append('Put `' + statements + '` inside this function if you want to reuse it.')
        if missing_helpers:
            hints.append('Make the helper dependencies retainable too: ' + ', '.join(missing_helpers) + '.')
        if missing_data:
            hints.append('Pass ' + ', '.join(missing_data) + ' as arguments; snippet variables are not retained.')
        item['hint'] = ' '.join(hints)
    return kept, rejected + invalid
