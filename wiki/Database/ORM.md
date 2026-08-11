# ORM

Object-Relational Mapping (ORM) provides a high-level interface for database operations.

## Usage

```bash
# Define model
aweai db orm define Model --table models --fields "name:TEXT,type:TEXT"

# Query with ORM
aweai db orm query Model --where "type=transformer"
```

```python
from aweai.database.orm import Model, Field

class Model(Model):
    name = Field(str)
    type = Field(str)
    params = Field(dict)

# Query
results = Model.query().where(type="transformer").all()
```

## Related Pages

- [Schema](Schema.md) — Schema
- [QueryBuilder](QueryBuilder.md) — Query builder
