# Quick Reference Card

## File Structure

```
app.py (92 lines) ← Main application entry point
│
├── DatabaseService
│   ├── database_service.py
│   ├── __init__(engine)
│   └── Methods:
│       ├── get_active_models_status()
│       ├── get_recent_forecasts()
│       ├── get_input_forecasts()
│       ├── get_recent_logs()
│       ├── get_historical_data()
│       ├── get_model_metrics()
│       ├── get_confidence()
│       └── get_all_logs()
│
├── layout_components
│   ├── layout_components.py
│   └── Functions:
│       ├── create_header()
│       ├── create_collapsible_section()
│       ├── create_model_monitoring_tab()
│       ├── create_system_logs_tab()
│       ├── create_model_metrics_tab()
│       └── create_status_table()
│
└── CallbackHandler
    ├── callback_handlers.py
    ├── __init__(app, db_service, debug_timestamp)
    ├── register_callbacks()
    └── Callback groups:
        ├── _register_tab_callbacks()
        ├── _register_status_callbacks()
        ├── _register_detail_callbacks()
        ├── _register_system_log_callbacks()
        └── _register_collapse_callbacks()
```

## Common Operations

### Starting the App
```bash
python app.py  # Runs on http://localhost:8050
```

### Adding a Database Query
1. Add method to `DatabaseService` in `database_service.py`
2. Use in callbacks via `self.db_service.your_method()`

### Adding a Callback
1. Add method to `CallbackHandler` in `callback_handlers.py`
2. Call from `register_callbacks()` method

### Adding UI Component
1. Create function in `layout_components.py`
2. Import and use in layout

### Debugging
1. Set `DEBUG = True` in `app.py`
2. Set `DEBUG_USE_STATIC_TIMESTAMP = True` for fixed time
3. Check logs in console output

## Import Statements

```python
# Get database service
from dashboard.database_service import DatabaseService

# Get layout builders
from dashboard.layout_components import (
    create_header,
    create_model_monitoring_tab,
)

# Get callback handler
from dashboard.callback_handlers import CallbackHandler

# Get database engine
import database.db_tools as dbt
```

## Type Hints Cheatsheet

```python
from typing import Optional, List, Dict, Tuple, Any
import pandas as pd
from dash import html

def example_function(
    model_id: int,                          # Required int
    name: Optional[str] = None,             # Optional string
    items: List[str] = None,                # List of strings
    config: Dict[str, Any] = None,          # Dictionary
    data: pd.DataFrame = None,              # Pandas DataFrame
) -> Tuple[html.Div, pd.DataFrame]:        # Returns tuple
    pass
```

## Docstring Template

```python
def my_function(param1: int, param2: str) -> Dict:
    """One-line summary.

    Longer description if needed.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When something is invalid
    """
    pass
```

## Key Classes

### DatabaseService
```python
service = DatabaseService(engine)
data = service.get_recent_forecasts(model_id=1)
logs = service.get_recent_logs(sensor_name="sensor1")
```

### CallbackHandler
```python
handler = CallbackHandler(app, db_service, debug_timestamp=None)
handler.register_callbacks()
```

### ForecastMonitor
```python
monitor = ForecastMonitor()
monitor.run(host="0.0.0.0", port=8050)
```

## Error Handling Pattern

```python
def get_data(self, id: int) -> pd.DataFrame:
    try:
        with Session(self.engine) as session:
            data = session.query(Table).filter(...).all()
            return pd.DataFrame([...])
    except Exception as e:
        logger.error("Error getting data: %s", str(e))
        return pd.DataFrame()
```

## Testing Pattern

```python
from unittest.mock import Mock
from dashboard.database_service import DatabaseService

# Mock the engine
mock_engine = Mock()

# Create service with mock
service = DatabaseService(mock_engine)

# Test the method
result = service.get_active_models_status()
assert result is not None
```

## Callback Pattern

```python
@self.app.callback(
    Output('output-id', 'property'),
    Input('input-id', 'property'),
    State('state-id', 'property')
)
def my_callback(input_value, state_value):
    # Get data
    data = self.db_service.get_data(input_value)

    # Format UI
    figure = create_plot(data)

    # Return
    return figure
```

## Layout Pattern

```python
def create_my_section() -> dbc.Card:
    return dbc.Card([
        dbc.CardHeader(html.H3("Title")),
        dbc.CardBody([
            html.Div(id="my-content")
        ])
    ])
```

## Useful Classes/Functions

| Import                  | Purpose               |
| ----------------------- | --------------------- |
| `DatabaseService`       | Database operations   |
| `CallbackHandler`       | Callback registration |
| `ForecastMonitor`       | Main application      |
| `create_header()`       | Header component      |
| `create_status_table()` | Status table          |
| `dbc.Card`              | Card container        |
| `dcc.Graph`             | Plotly graph          |
| `dash_table.DataTable`  | Data table            |

## Environment Variables

Set in `.env` or environment:
- `DB_KIND` - "oracle" or "sqlite"
- `DB_USER` - Database username
- `DB_PASSWORD` - Database password
- `DB_DSN` - Database connection string
- `lib_dir` - Oracle client library path (optional)

## Configuration Constants

Edit in `dashboard/constants.py`:
- `NUM_RECENT_FORECASTS` - Forecasts to display
- `NUM_RECENT_LOGS` - Logs to display
- `BUFFER_TIME` - Time buffer for queries
- `LOOKBACK_EXTERNAL_FORECAST_EXISTENCE` - Days to look back

## Debug Tips

```python
# Enable all debug output
DEBUG = True

# Use fixed timestamp for testing
DEBUG_USE_STATIC_TIMESTAMP = True
DEBUG_TIMESTAMP = pd.to_datetime("2024-09-13 14:00:00.000")

# Check logs
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Tips

1. **Use debug timestamp** for consistent testing
2. **Limit query results** with LIMIT/OFFSET
3. **Cache frequently accessed data** with decorators
4. **Lazy load** data only when needed
5. **Paginate large datasets** using DataTable

## Documentation Files

- `REFACTORING_COMPLETE.md` - Overall summary
- `REFACTORING_SUMMARY.md` - Detailed changes
- `ARCHITECTURE.md` - System design
- `DEVELOPER_GUIDE.md` - Extension guide
- `BEFORE_AFTER.md` - Code examples

---

**Need help?** Check the documentation files or look at the source code comments.
