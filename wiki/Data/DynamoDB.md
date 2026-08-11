# DynamoDB

Amazon DynamoDB NoSQL database support.

## Usage

```python
from aweai.data.formats import DynamoDBFormat

dynamodb = DynamoDBFormat(region="us-east-1")
item = dynamodb.get_item("table_name", key={"id": "123"})
```

## Related Pages

- [HBase](HBase.md) — HBase
- [Bigtable](Bigtable.md) — Bigtable
- [Formats](Formats.md) — Data formats
