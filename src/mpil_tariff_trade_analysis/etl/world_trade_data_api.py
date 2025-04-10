### Custom implementation of the world_trade_data (WITS) API, given the python package doesn't work.

"""WITS Data: indicators, tariffs, and referential data"""

import pandas as pd
import requests
import xmltodict

from mpil_tariff_trade_analysis.utils.logging_config import get_logger

# Replace the basic logging setup with our configured logger
LOGGER = get_logger(__name__)

"""Default values for the WITS requests"""
DEFAULT_YEAR = '2017'
DEFAULT_DATASOURCE = 'tradestats-trade'

LIMITATIONS = """Please read the **Limitation on Data Request** from https://wits.worldbank.org/witsapiintro.aspx

> In order to avoid server overload, request for entire database is not possible in one query.
> The following are the options to request data:
> - Maximum of two dimension with 'All' value is allowed.
> - Data request with All Reporter and All Partners is not allowed.
> - When two of the dimensions have 'All', then rest of the dimension should have a specific value.
> - Trade Stats data for a single reporter, partner, Indicator and product can be requested for a 
range of years."""

DATASOURCES = ["trn", "tradestats-trade", "tradestats-tariff"]


# Helper functions
def semicolon_separated_strings(value):
    """Turn lists into semicolon separated strings"""
    if isinstance(value, list):
        return ";".join(semicolon_separated_strings(v) for v in value)
    return str(value)


def true_or_false(value):
    """Replace Yes/No with True/False"""
    if value in {'0', 'Yes'}:
        return True
    if value in {'1', 'No'}:
        return False
    raise ValueError('{} is neither True nor False'.format(value))


# Tariff functions
def get_tariff_reported(
    reporter,
    partner="000",
    product="all",
    year=DEFAULT_YEAR,
    name_or_id="name",
):
    """Tariffs (reported)"""
    return _get_data(
        reporter,
        partner,
        product,
        year,
        is_tariff=True,
        datatype="reported",
        datasource="trn",
        name_or_id=name_or_id,
    )


def get_tariff_estimated(
    reporter,
    partner="000",
    product="all",
    year=DEFAULT_YEAR,
    name_or_id="name",
):
    """Tariffs (estimated)"""
    return _get_data(
        reporter,
        partner,
        product,
        year,
        is_tariff=True,
        datatype="aveestimated",
        datasource="trn",
        name_or_id=name_or_id,
    )


def get_indicator(
    indicator,
    reporter,
    partner="wld",
    product="all",
    year=DEFAULT_YEAR,
    datasource=DEFAULT_DATASOURCE,
    name_or_id="name",
):
    """Get the values for the desired indicator"""
    return _get_data(
        reporter,
        partner,
        product,
        year,
        indicator=indicator,
        datasource=datasource,
        name_or_id=name_or_id,
    )


def _get_data(reporter, partner, product, year, datasource, name_or_id, is_tariff=False, **kwargs):
    args = {
        "reporter": reporter,
        "partner": partner,
        "product": product,
        "year": year,
        "datasource": datasource,
    }
    args.update(kwargs)
    list_args = []

    # Format the arguments
    if datasource == "trn":
        order = ["datasource", "reporter", "partner", "product", "year"]
    else:
        order = ["datasource", "reporter", "year", "partner", "product"]
    for arg in order + list(kwargs.keys()):
        list_args.append(arg)
        list_args.append(semicolon_separated_strings(args[arg]))

    # Check we're not asking for too many 'alls'
    if ("all" in reporter.lower() and "all" in partner.lower()) or len(
        [k for k in args if "all" in k.lower()]
    ) >= 3:
        LOGGER.warning(LIMITATIONS)

    response = requests.get(
        "https://wits.worldbank.org/API/V1/SDMX/V21/{}?format=JSON".format("/".join(list_args))
    )
    response.raise_for_status()
    data = response.json()
    df = _wits_data_to_df(data, name_or_id=name_or_id, is_tariff=is_tariff)
    if is_tariff and not len(df):
        LOGGER.warning("""
                       Did you know? The reporter-partner combination only yields results
                       if the two countries have a preferential trade agreement (PTA).
                       Otherwise, all other tariffs to all non-PTA countries
                       are found if one enters "000" in partner.
                       """)
    return df


def _wits_data_to_df(data, value_name="Value", is_tariff=False, name_or_id="id"):
    observation = data["structure"]["attributes"]["observation"]
    levels = data["structure"]["dimensions"]["series"]
    obs_levels = data["structure"]["dimensions"]["observation"]
    series = data["dataSets"][0]["series"]

    index_names = [level["name"] for level in levels] + [
        obs_level["name"] for obs_level in obs_levels
    ]
    column_names = [value_name] + [o["name"] for o in observation]

    all_observations = {value_name: []}
    for col in index_names:
        all_observations[col] = []
    for col in column_names:
        all_observations[col] = []

    for i in series:
        loc = [int(j) for j in i.split(":")]

        # When loading tariffs, product is at depth 3, but levels say it's at depth 4
        # - So we invert the two levels
        if is_tariff:
            loc[2], loc[3] = loc[3], loc[2]

        observations = series[i]["observations"]
        for obs in observations:
            for level, j in zip(levels, loc, strict=False):
                all_observations[level["name"]].append(level["values"][j][name_or_id])

            o_loc = [int(j) for j in obs.split(":")]
            for level, j in zip(obs_levels, o_loc, strict=False):
                all_observations[level["name"]].append(level["values"][j]["name"])

            values = observations[obs]
            all_observations[value_name].append(float(values[0]))
            for obs_ref, value in zip(observation, values[1:], strict=False):
                if isinstance(value, int) and len(obs_ref["values"]) > value:
                    all_observations[obs_ref["name"]].append(obs_ref["values"][value][name_or_id])
                else:
                    all_observations[obs_ref["name"]].append(value)

    table = pd.DataFrame(all_observations).set_index(index_names)[column_names]
    for col in ["NomenCode", "TariffType", "OBS_VALUE_MEASURE"]:
        if col in table:
            table[col] = table[col].astype("category")

    for col in table:
        if "_Rate" in col or "Lines" in col or col == "Value":
            table[col] = table[col].apply(lambda s: pd.np.NaN if s == "" else float(s))

    return table


# Referential data functions
def get_referential(name, datasource=DEFAULT_DATASOURCE):
    """Return the desired referential"""
    LOGGER.info(f"Getting referential data for {name} from datasource {datasource}")
    args = ['datasource', datasource, name]
    response = requests.get('https://wits.worldbank.org/API/V1/wits/{}/'.format('/'.join(args)))
    response.raise_for_status()
    data_dict = xmltodict.parse(response.content)

    if 'wits:error' in data_dict:
        if name == 'indicator' and datasource == 'trn':
            msg = "No indicator is available on datasource='trn'. " \
                  "Please use either {}".format(' or '.join("datasource='{}'".format(src)
                                                            for src in DATASOURCES if
                                                            src != datasource))
        else:
            msg = data_dict['wits:error']['wits:message']['#text']
        LOGGER.error(f"Error getting referential data: {msg}")
        raise ValueError(msg)

    def deeper(key, ignore_if_missing=False):
        if key not in data_dict:
            if ignore_if_missing:
                return data_dict
            error_msg = f'{key} not in {data_dict.keys()}'
            LOGGER.error(error_msg)
            raise KeyError(error_msg)
        return data_dict[key]

    if name == 'country':
        level1 = 'countries'
        level2 = name
    elif name == 'dataavailability':
        level1 = name
        level2 = 'reporter'
    else:
        level1 = name + 's'
        level2 = name

    data_dict = deeper('wits:datasource')
    data_dict = deeper('wits:{}'.format(level1))
    data_dict = deeper('wits:{}'.format(level2))

    for obs in data_dict:
        if 'wits:reporternernomenclature' in obs:
            obs['wits:reporternernomenclature'] = obs['wits:reporternernomenclature']['@reporternernomenclaturecode']

    table = pd.DataFrame(data_dict)
    # Clean up column names by removing special characters
    table.columns = [
        col.replace('@', '').replace('#', '').replace('wits:', '') 
        for col in table.columns
    ]

    for col in table:
        if col == 'notes':
            table[col] = table[col].apply(lambda note: '' if note is None else note)
        if col.startswith('is') and not col.startswith('iso'):
            try:
                table[col] = table[col].apply(true_or_false)
            except ValueError:
                pass

    LOGGER.info(f"Successfully retrieved {len(table)} {name} records")
    return table


def get_countries(datasource=DEFAULT_DATASOURCE):
    """List of countries for the given datasource"""
    LOGGER.info(f"Getting countries list from datasource {datasource}")
    table = get_referential('country', datasource=datasource)
    table = table.set_index('iso3Code')[
        ['name', 'notes', 'countrycode', 'isreporter', 'ispartner', 'isgroup', 'grouptype']]

    return table


def get_nomenclatures(datasource=DEFAULT_DATASOURCE):
    """List of nomenclatures for the given datasource"""
    LOGGER.info(f"Getting nomenclatures from datasource {datasource}")
    table = get_referential('nomenclature', datasource=datasource)
    return table.set_index('nomenclaturecode')[['text', 'description']]


def get_products(datasource=DEFAULT_DATASOURCE):
    """List of products for the given datasource"""
    LOGGER.info(f"Getting products list from datasource {datasource}")
    table = get_referential('product', datasource=datasource)
    return table.set_index('productcode')


def get_dataavailability(datasource=DEFAULT_DATASOURCE):
    """Data availability for the given datasource"""
    LOGGER.info(f"Getting data availability from datasource {datasource}")
    table = get_referential('dataavailability', datasource=datasource)
    return table.set_index(['iso3Code', 'year']).sort_index()


def get_indicators(datasource=DEFAULT_DATASOURCE):
    """List of indicators for the given datasource"""
    LOGGER.info(f"Getting indicators list from datasource {datasource}")
    table = get_referential('indicator', datasource=datasource)
    return table.set_index(['indicatorcode']).sort_index()


if __name__ == "__main__":
    # Test the tariff endpoint
    LOGGER.info("Testing tariff endpoint...")
    tariff_data = get_tariff_reported(reporter="840", partner="000", product="020110", year='all')
    LOGGER.info(f"Retrieved {len(tariff_data)} tariff records")
    LOGGER.info(tariff_data.head(2))
    
    # Test the referential data endpoint
    LOGGER.info("Testing referential data endpoints...")
    countries = get_countries()
    LOGGER.info(f"Retrieved {len(countries)} countries")
    LOGGER.info(countries.head(2))