import pandas as pd


class CountryCodeConverter:
    def __init__(self):
        df = pd.read_csv("data/raw/BACI_HS92_V202501/country_codes_V202501.csv")
        self.df = df
        self.columns = df.columns.tolist()

    def convert(self, value, from_col, to_col):
        if from_col not in self.columns or to_col not in self.columns:
            raise ValueError(f"Invalid column name(s). Choose from: {self.columns}")

        match = self.df[self.df[from_col] == value]
        if match.empty:
            raise ValueError(f"No match found for {value} in column {from_col}")
        return match.iloc[0][to_col]


def agg_harmonized_system_code(hs_code, agg_level="2digit"):
    if agg_level not in ["2digit", "4digit", "6digit"]:
        raise ValueError("Invalid aggregation level. Choose from: ['2digit', '4digit', '6digit']")

    if agg_level == "2digit":
        return hs_code[:2]
    elif agg_level == "4digit":
        return hs_code[:4]
    else:
        return hs_code[:6]


if __name__ == "__main__":
    converter = CountryCodeConverter()
    print(converter.convert("USA", "country_name", "country_code"))

    print(agg_harmonized_system_code("010110", "2digit"))
    print(agg_harmonized_system_code("010110", "4digit"))
    print(agg_harmonized_system_code("010110", "6digit"))
