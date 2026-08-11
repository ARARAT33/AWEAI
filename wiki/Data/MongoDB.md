# MongoDB

MongoDB document database support.

## Usage

```python
from aweai.data.formats import MongoDBFormat

mongo = MongoDBFormat(uri="mongodb://localhost:27017")
data = mongo.find("collection", query={})
```

## Related Pages

- [MySQL](MySQL.md) — MySQL
- [Redis](Redis.md) — Redis
- [Formats](Formats.md) — Data formats
