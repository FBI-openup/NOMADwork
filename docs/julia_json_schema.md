# Julia Solver JSON Schema

The Julia implementation relies on JSON inputs that mirror the canonical `.in` challenge files with explicit field names. The helper script `python_tools/export_instances.py` converts existing datasets.

## Input payload
```jsonc
{
  "grid": {
    "rows": <int>,     // M dimension
    "cols": <int>,     // N dimension
    "horizon": <int>   // total number of seconds T
  },
  "uavs": [
    {
      "x": <int>,
      "y": <int>,
      "peak_bandwidth": <float>, // B parameter
      "phase": <int>             // I+ parameter in slots
    }
  ],
  "flows": [
    {
      "id": <int>,
      "access": {"x": <int>, "y": <int>},
      "start_time": <int>,    // flow release time t_start
      "volume": <float>,      // Q_total in Mbits
      "zone": {
        "x_min": <int>, "y_min": <int>,
        "x_max": <int>, "y_max": <int>
      }
    }
  ]
}
```

This structure preserves every requirement from `problem_definition.md`: the grid topology, per-UAV bandwidth pattern, and each flow's admissible landing rectangle.

## Output payload
```jsonc
{
  "assignments": [
    {
      "flow_id": <int>,
      "segments": [
        {"time": <int>, "x": <int>, "y": <int>, "rate": <float>}
      ]
    }
  ],
  "score": {
    "total": <float>,  // aggregate challenge score
    "by_flow": [
      {
        "flow_id": <int>,
        "traffic": <float>,
        "delay": <float>,
        "distance": <float>,
        "landing": <float>,
        "weight": <float>
      }
    ]
  }
}
```

The Julia CLI converts this JSON response back to the standard `.out` format when required, ensuring interoperability with the existing Python pipeline and scoring utilities.
