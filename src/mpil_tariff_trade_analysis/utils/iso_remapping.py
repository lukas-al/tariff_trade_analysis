"""Remapping ISO codes to a shared space"""

# Manually processed and structured data from the provided text
# Note: This required significant interpretation due to formatting issues in the raw text.
# Missing values are represented by None or ''.

from mpil_tariff_trade_analysis.utils.worldbase_2_iso import (
    COUNTRY_MAPPING_DATA,
    _iso_2char_map,
    _iso_3char_map,
    _iso_3digit_map,
    _iso_name_map,
    _wb_2char_map,
    _wb_3digit_map,
    _wb_name_map,
)


def get_country_mapping(value, input_type, output_type):
    """
    Maps country identifiers between WorldBase (WB) and ISO standards.

    Args:
        value: The identifier value to map (e.g., '777', '826', 'Germany', 'DEU').
        input_type (str): The type of the input value. Valid types are:
            'wb_3digit', 'wb_2char', 'wb_name',
            'iso_3digit', 'iso_3char', 'iso_2char', 'iso_name'.
        output_type (str): The desired output field type. Valid types are:
            'wb_3digit', 'wb_2char', 'wb_name',
            'iso_3digit', 'iso_3char', 'iso_2char', 'iso_name',
            'full_record' (returns the entire dictionary).

    Returns:
        The mapped value (str or dict) if found, otherwise None.
        For 'full_record', returns the dictionary associated with the found entry.
    """
    if not value or not input_type or not output_type:
        return None

    # Normalize input for lookup (uppercase for names/chars, keep digits as string)
    value_str = str(value)
    lookup_value = value_str.upper() if "name" in input_type or "char" in input_type else value_str
    input_type = input_type.lower()
    output_type = output_type.lower()

    idx = None
    # Find index using the appropriate map
    if input_type == "wb_3digit":
        idx = _wb_3digit_map.get(lookup_value)
    elif input_type == "wb_2char":
        idx = _wb_2char_map.get(lookup_value)
    elif input_type == "wb_name":
        idx = _wb_name_map.get(lookup_value)
    elif input_type == "iso_3digit":
        idx = _iso_3digit_map.get(lookup_value)
    elif input_type == "iso_3char":
        idx = _iso_3char_map.get(lookup_value)
    elif input_type == "iso_2char":
        idx = _iso_2char_map.get(lookup_value)
    elif input_type == "iso_name":
        idx = _iso_name_map.get(lookup_value)
    else:
        print(f"Error: Invalid input_type '{input_type}'")
        return None  # Invalid input type

    if idx is not None:
        record = COUNTRY_MAPPING_DATA[idx]
        if output_type == "full_record":
            return record.copy()  # Return a copy of the full record
        elif output_type in record:
            return record.get(output_type)
        else:
            print(f"Error: Invalid output_type '{output_type}' or missing in record")
            return None  # Invalid output type or field missing in found record
    else:
        return None  # Input value not found


# --- Examples ---

if __name__ == "__main__":
    print("--- WB 3 digit to ISO 3 digit ---")
    print(f"WB '777' (UAE) -> ISO 3d: {get_country_mapping('777', 'wb_3digit', 'iso_3digit')}")
    print(
        f"WB '898' (Ajman) -> ISO 3d: {get_country_mapping('898', 'wb_3digit', 'iso_3digit')}"
    )  # Maps to UAE ISO
    print(f"WB '805' (USA) -> ISO 3d: {get_country_mapping('805', 'wb_3digit', 'iso_3digit')}")
    print(f"WB '034' (Aruba) -> ISO 3d: {get_country_mapping('034', 'wb_3digit', 'iso_3digit')}")

    print("\n--- ISO 3 digit to WB 3 digit ---")
    print(
        f"ISO '784' (UAE) -> WB 3d: {get_country_mapping('784', 'iso_3digit', 'wb_3digit')}"
    )  # Maps to primary UAE WB code
    print(
        f"ISO '826' (UK) -> WB 3d: {get_country_mapping('826', 'iso_3digit', 'wb_3digit')}"
    )  # Maps to primary UK WB code (which was missing, so None)
    print(f"ISO '840' (USA) -> WB 3d: {get_country_mapping('840', 'iso_3digit', 'wb_3digit')}")
    print(f"ISO '533' (Aruba) -> WB 3d: {get_country_mapping('533', 'iso_3digit', 'wb_3digit')}")

    print("\n--- Other Mappings ---")
    print(
        f"WB Name 'Germany' -> ISO 3 Char: {get_country_mapping('Germany', 'wb_name', 'iso_3char')}"
    )
    print(f"ISO 3 Char 'DEU' -> WB Name: {get_country_mapping('DEU', 'iso_3char', 'wb_name')}")
    print(
        f"ISO 2 Char 'GB' -> WB Name: {get_country_mapping('GB', 'iso_2char', 'wb_name')}"
    )  # Should map to 'United Kingdom'
    print(
        f"WB 3 digit '785' (England) -> ISO Name: {get_country_mapping('785', 'wb_3digit', 'iso_name')}"
    )
    print(
        f"ISO 3 digit '826' (UK) -> Full Record: {get_country_mapping('826', 'iso_3digit', 'full_record')}"
    )
    print(
        f"WB 3 digit '999' (Non-existent) -> ISO 3d: {get_country_mapping('999', 'wb_3digit', 'iso_3digit')}"
    )
