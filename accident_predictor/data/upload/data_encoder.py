import numpy as np


def sin_cos_representation(data, value_range):
    """
    :param data: data value
    :param value_range: what is the data range of the cycle
    :return: sin and cos values for the data value
    """
    sin_value = (np.sin(2*np.pi*float(data)/value_range) + 1) / 2
    cos_value = (np.cos(2*np.pi*float(data)/value_range) + 1) / 2
    return sin_value, cos_value


def time_encoder(raw_time):
    """
    Convert the raw_time format from the csv file (0-2400) converted to minutes in a day (24h * 60m)
    :param raw_time: time in format of csv file, represented as integer [0, 2400] e.g. 1530 for 15:30
    :return: encoded time in minutes range (0-1440) sin and cos representation
    """
    tmp_minutes = int(raw_time) % 100
    hours = (int(raw_time) - tmp_minutes) / 100
    minutes = hours*60 + tmp_minutes
    sin, cos = sin_cos_representation(minutes, 1440)
    return minutes, sin, cos


def date_encoder(raw_date):
    """
    Convert the raw_date format from the csv file to date in range of 0-361, a month is averaged to 30 days
    :param raw_date: date in format of csv file e.g. 8. Dez. or 29-May-10
    :return: encoded date in format 0-365 in sin and cos representation
    """

    month_map = {
        0: ["Jan", "Januar", "January"],
        1: ["Feb", "Februar", "February"],
        2: ["Mär", "Mar", "Mrz", "März", "March"],
        3: ["Apr", "April"],
        4: ["May", "Mai"],
        5: ["June", "Juni"],
        6: ["July", "Juli", "Jul"],
        7: ["Aug", "August"],
        8: ["Sep", "September"],
        9: ["Oct", "October", "Oktober", "Okt"],
        10: ["Nov", "November"],
        11: ["Dec", "Dez", "December", "Dezember"]
    }

    def get_month_nr(search_str):
        for i, month_str_array in month_map.items():
            if any(search_str.lower().strip() in s.lower() for s in month_str_array):
                return i
        raise ValueError("ENCODING ERROR: Unknown date string " + search_str)

    splited = raw_date.split("-")

    if len(splited) > 1:
        # got date of format 29-May-10
        month_nr = get_month_nr(splited[1])
        int_value = (month_nr * 30) + int(splited[0])
    else:
        splited = raw_date.split(".")
        if len(splited) > 1:
            # got date of format 8. Dez.
            month_nr = get_month_nr(splited[1])
            int_value = (month_nr * 30) + int(splited[0])
        else:
            raise ValueError("ENCODING ERROR: Unknown date format " + raw_date)

    sin_value, cos_value = sin_cos_representation(int_value, 361)
    return int_value, sin_value, cos_value


def class_encoder(class_raw):
    """
    Convert accident class (string) into a encoded array
    :param class_raw: accident class as a string
    :return: 1-hot array encoding the class
    """

    class_map = ["Fahrer", "Passagier", "Fussgänger"]
    class_array = [0, 0, 0]

    for i, class_name in enumerate(class_map):
        if class_raw.strip().lower() == class_name.lower():
            class_array[i] = 1
            return class_array
    raise ValueError("ENCODING ERROR: Unknown class " + class_raw)


def light_encoder(light_raw):
    """
    Convert the light class (string) into a encoded array
    :param light_raw: string of the light class
    :return: 1-hot array encoding the light class
    """
    class_map = [
        "Tageslicht: Strassenbeleuchtung vorhanden",
        "Dunkelheit: Strassenbeleuchtung vorhanden und beleuchtet",
        ["Dunkelheit: keine Strassenbeleuchtung", "Dunkelheit: Strassenbeleuchtung vorhanden und nicht beleuchtet"],
        "Dunkelheit: Strassenbeleuchtung unbekannt"
    ]
    class_array = [0, 0, 0, 0]

    for i, class_name in enumerate(class_map):
        found_str = False
        if isinstance(class_name, list):
            for n in class_name:
                if light_raw.strip().lower() == n.lower():
                    found_str = True
        else:
            found_str = light_raw.strip().lower() == class_name.lower()

        if found_str:
            class_array[i] = 1
            return class_array
    raise ValueError("ENCODING ERROR: Unknown light class " + light_raw)


def ground_encoder(ground_raw):
    """
    Convert ground condition description into 1-hot encoded array
    :param ground_raw: string describing ground condition
    :return: 1-hot array encoding the ground condition
    """
    class_map = ["trocken", "nass / feucht", ["Schnee", "Frost/ Ice", "Frost / Eis"], ["unkown", "9", "Überflutung"]]
    class_array = [0, 0, 0, 0]

    for i, class_name in enumerate(class_map):
        found_str = False
        if isinstance(class_name, list):
            for n in class_name:
                if ground_raw.strip().lower() == n.lower():
                    found_str = True
        else:
            found_str = ground_raw.strip().lower() == class_name.lower()

        if found_str:
            class_array[i] = 1
            return class_array
    raise ValueError("ENCODING ERROR: Unknown ground condition " + ground_raw)


def gender_encoder(gender_raw):
    """
    Convert the gender (string) of the driver into a encoded array
    :param gender_raw: string of the gender of the driver
    :return: 1-hot array encoding the gender of the driver
    """
    class_map = ["weiblich", "männlich"]
    class_array = [0, 0]

    for i, class_name in enumerate(class_map):
        if gender_raw.strip().lower() == class_name.lower():
            class_array[i] = 1
            return class_array
    raise ValueError("ENCODING ERROR: Unknown gender " + gender_raw)


def vehicle_encoder(vehicle_raw):
    """
    Convert the vehicle type (string) into a encoded array
    :param vehicle_raw: string of the vehicle
    :return: 1-hot array encoding the vehicle type
    """
    class_map = ["Auto", "Taxi", "LKW ab 7.5t", "Fahrrad", ["Mottorrad (125cc)", "Mottorrad"],
                 ["Kleinbus", "Transporter", "LKW bis 7.5t"],
                 "Mottorrad (50cc)", "Mottorrad (500cc)",
                 "Bus", ["Unbekannt", "Traktor", "Pferd", "Andere", "97"]]
    class_array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for i, class_name in enumerate(class_map):
        found_str = False
        if isinstance(class_name, list):
            for n in class_name:
                if vehicle_raw.strip().lower() == n.lower():
                    found_str = True
        else:
            found_str = vehicle_raw.strip().lower() == class_name.lower()

        if found_str:
            class_array[i] = 1
            return class_array
    raise ValueError("ENCODING ERROR: Unknown vehicle type " + vehicle_raw)


def weather_encoder(weather_raw):
    """
    Convert the weather (string) into a encoded array
    :param weather_raw: string that describes the weather
    :return: 1-hot array encoding weather
    """
    class_map = [["Gut", "Gut (starker Wind)"], ["Regen", "Regen (starker Wind)"],
                 ["Schnee", "Schnee (starker Wind)"], "Nebel",
                 ["Unbekannt", "Andere"]]
    class_array = [0, 0, 0, 0, 0]

    for i, class_name in enumerate(class_map):
        found_str = False
        if isinstance(class_name, list):
            for n in class_name:
                if weather_raw.strip().lower() == n.lower():
                    found_str = True
        else:
            found_str = weather_raw.strip().lower() == class_name.lower()

        if found_str:
            class_array[i] = 1
            return class_array
    raise ValueError("ENCODING ERROR: Unknown weather condition " + weather_raw)


def road_encoder(road_type_raw):
    """
    Convert the road type (string) of a encoded array
    :param road_type_raw: string of the road type
    :return: 1-hot array encoding the road type
    """
    class_map = ["Bundesstrasse", "Autobahn", "Landesstrasse", "Kraftfahrzeugstrasse",
                 ["nicht klassifiziert", "unbefestigte Strasse"]]
    class_array = [0, 0, 0, 0, 0, 0]

    for i, class_name in enumerate(class_map):
        found_str = False
        if isinstance(class_name, list):
            for n in class_name:
                if road_type_raw.strip().lower() == n.lower():
                    found_str = True
        else:
            found_str = road_type_raw.strip().lower() == class_name.lower()

        if found_str:
            class_array[i] = 1
            return class_array
    raise ValueError("ENCODING ERROR: Unknown road type " + road_type_raw)

