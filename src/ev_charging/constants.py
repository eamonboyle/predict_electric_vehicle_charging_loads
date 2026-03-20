TRAFFIC_COUNT_COLUMNS = [
    "KROPPAN BRU",
    "MOHOLTLIA",
    "SELSBAKK",
    "MOHOLT RAMPE 2",
    "Jonsvannsveien vest for Steinanvegen",
]

EV_REQUIRED_COLUMNS = [
    "session_ID",
    "Garage_ID",
    "User_ID",
    "User_type",
    "Shared_ID",
    "Start_plugin",
    "Start_plugin_hour",
    "End_plugout",
    "End_plugout_hour",
    "El_kWh",
    "Duration_hours",
    "month_plugin",
    "weekdays_plugin",
    "Plugin_category",
    "Duration_category",
]

TRAFFIC_REQUIRED_COLUMNS = ["Date_from", "Date_to", *TRAFFIC_COUNT_COLUMNS]

META_COLUMNS_FOR_SPLIT = ["session_ID", "User_ID", "Garage_ID"]

DROP_FOR_FEATURES = [
    "session_ID",
    "Garage_ID",
    "User_ID",
    "Shared_ID",
    "Plugin_category",
    "Duration_category",
    "Start_plugin",
    "Start_plugin_hour",
    "End_plugout",
    "End_plugout_hour",
    "Date_from",
    "Date_to",
]
