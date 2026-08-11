# Schema

Schema management handles table definitions, constraints, and migrations.

## Usage

```bash
# Create table
aweai db schema create-table models --columns "id SERIAL, name TEXT, type TEXT"

# Alter table
aweai db schema alter-table models --add-column params JSONB
```

```python
from aweai.database.schema import SchemaManager

sm = SchemaManager()
sm.create_table("models", columns=["id SERIAL", "name TEXT", "type TEXT"])
sm.add_column("models", "params", "JSONB")
```

## Related Pages

- [Migration](Migration.md) — Migration
- [Query](Query.md) — Query
