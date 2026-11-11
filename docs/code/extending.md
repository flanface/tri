# 9. Extending

- **New condition kind:** add it to `Condition` and `holds` (`tri/rules/defs.py`), give `Rulebook.mine` (`tri/rules/passes.py`) a feature column for it, and describe it in `Condition.text`.
- **New context field** (for example bell): add it to `TriConfig.context_fields` (`tri/engine/defs.py`); it becomes a one-hot in the field and a context atom in rules.
- **New review reason:** add it to `types.Reason`, map it in `induction.RuleInducer` and in `authority` binding.
- **Different data source:** produce `Build` objects (or `CompactBuilds`) and call `Tri.fit`. Nothing else depends on the planner format.
- **New definition or method module:** put value types and configurations in the subpackage's `defs.py`; put algorithms in a sibling module. A new `Tri`, `PassEngine` or `Catalog` method goes in the mixin module for its concern ([Section 1](layout.md)).
