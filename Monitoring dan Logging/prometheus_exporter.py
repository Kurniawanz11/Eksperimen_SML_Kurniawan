from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "request_count",
    "Jumlah request API"
)

PREDICTION_COUNT = Counter(
    "prediction_count",
    "Jumlah prediksi model"
)

REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "Waktu respon API"
)