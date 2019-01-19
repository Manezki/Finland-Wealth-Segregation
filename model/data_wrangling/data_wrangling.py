import pandas as pd
import numpy as np
import pickle
from os import path as op

PAAVO_PATH = op.join(op.dirname(__file__), '..', '..', 'data', 'paavo_9_koko.csv')
column_map = {
    'Postal code area' : '',
    'X coordinate in metres' : 'x',
    'Y coordinate in metres' : 'y',
    'Surface area' : 'surface_area',
    'Inhabitants, total, 2016 (HE)' : '',
    'Females, 2016 (HE)' : 'n_females_2016',
    'Males, 2016 (HE)' : 'n_males_2016',
    'Average age of inhabitants, 2016 (HE)' : 'avg_age_2016',
    '0-2 years, 2016 (HE)' : '',
    '3-6 years, 2016 (HE)' : '',
    '7-12 years, 2016 (HE)' : '',
    '13-15 years, 2016 (HE)' : '',
    '16-17 years, 2016 (HE)' : '',
    '18-19 years, 2016 (HE)' : '',
    '20-24 years, 2016 (HE)' : '',
    '25-29 years, 2016 (HE)' : '',
    '30-34 years, 2016 (HE)' : '',
    '35-39 years, 2016 (HE)' : '',
    '40-44 years, 2016 (HE)' : '',
    '45-49 years, 2016 (HE)' : '',
    '50-54 years, 2016 (HE)' : '',
    '55-59 years, 2016 (HE)' : '',
    '60-64 years, 2016 (HE)' : '',
    '65-69 years, 2016 (HE)' : '',
    '70-74 years, 2016 (HE)' : '',
    '75-79 years, 2016 (HE)' : '',
    '80-84 years, 2016 (HE)' : '',
    '85 years or over, 2016 (HE)' : '',
    'Aged 18 or over, total, 2016 (KO)' : '',
    'Basic level studies, 2016 (KO)' : '',
    'With education, total, 2016 (KO)' : '',
    'Matriculation examination, 2016 (KO)' : '',
    'Vocational diploma, 2016 (KO)' : '',
    'Academic degree - Lower level university degree, 2016 (KO)' : 'academic_degree_lower_2016',
    'Academic degree - Higher level university degree, 2016 (KO)' : 'academic_degree_higher_2016',
    'Aged 18 or over, total, 2015 (HR)' : '',
    'Average income of inhabitants, 2015 (HR)' : 'avg_inhabitant_income_2015',
    'Median income of inhabitants, 2015 (HR)' : 'median_inhabitant_income_2015',
    'Inhabintants belonging to the lowest income category, 2015 (HR)' : 'n_inhabitants_lowest_income_2015',
    'Inhabitants belonging to the middle income category, 2015 (HR)' : 'n_inhabitants_middle_income_2015',
    'Inhabintants belonging to the highest income category, 2015 (HR)' : 'n_inhabitants_highest_income_2015',
    'Accumulated purchasing power of inhabitants, 2015 (HR)' : 'acc_inhabitant_purchasing_power_2015',
    'Households, total, 2016 (TE)' : 'n_households_2016',
    'Average size of households, 2016 (TE)' : '',
    'Occupancy rate, 2016 (TE)' : '',
    'Young single persons, 2016 (TE)' : '',
    'Young couples without children, 2016 (TE)' : '',
    'Households with children, 2016 (TE)' : '',
    'Households with small children, 2016 (TE)' : '',
    'Households with children under school age, 2016 (TE)' : '',
    'Households with school-age children, 2016 (TE)' : '',
    'Households with teenagers, 2016 (TE)' : '',
    'Adult households, 2016 (TE)' : '',
    'Pensioner households, 2016 (TE)' : '',
    'Households living in owner-occupied dwellings, 2016 (TE)' : '',
    'Households living in rented dwellings, 2016 (TE)' : '',
    'Households living in other dwellings, 2016 (TE)' : '',
    'Households, total, 2015 (TR)' : 'n_households_2015',
    'Average income of households, 2015 (TR)' : 'avg_household_income_2015',
    'Median income of households, 2015 (TR)' : 'median_household_income_2015',
    'Households belonging to the lowest income category, 2015 (TR)' : 'n_households_lowest_income_2015',
    'Households belonging to the middle income category, 2015 (TR)' : 'n_households_middle_income_2015',
    'Households belonging to the highest income category, 2015 (TR)' : 'n_households_highest_income_2015',
    'Accumulated purchasing power of households, 2015 (TR)' : 'acc_household_purchasing_power_2015',
    'Free-time residences, 2016 (RA)' : '',
    'Buildings, total, 2016 (RA)' : '',
    'Other buildings, 2016 (RA)' : '',
    'Residential buildings, 2016 (RA)' : '',
    'Dwellings, 2016 (RA)' : 'n_dwellings_2016',
    'Average floor area, 2016 (RA)' : 'avg_floor_area_2016',
    'Dwellings in small houses, 2016 (RA)' : '',
    'Dwellings in blocks of flats, 2016 (RA)' : '',
    'Workplaces, 2015 (TP)' : '',
    'Primary production, 2015 (TP)' : '',
    'Processing, 2015 (TP)' : '',
    'Services, 2015 (TP)' : '',
    'A Agriculture, forestry and fishing, 2015 (TP)' : '',
    'B Mining and quarrying, 2015 (TP)' : '',
    'C Manufacturing, 2015 (TP)' : '',
    'D Electricity, gas, steam and air conditioning supply, 2015 (TP)' : '',
    'E Water supply; sewerage, waste management and remediation activities, 2015 (TP)' : '',
    'F Construction, 2015 (TP)' : '',
    'G Wholesale and retail trade; repair of motor vehicles and motorcycles, 2015 (TP)' : '',
    'H Transportation and storage, 2015 (TP)' : '',
    'I Accommodation and food service activities, 2015 (TP)' : '',
    'J Information and communication, 2015 (TP)' : '',
    'K Financial and insurance activities, 2015 (TP)' : '',
    'L Real estate activities, 2015 (TP)' : '',
    'M Professional, scientific and technical activities, 2015 (TP)' : '',
    'N Administrative and support service activities, 2015 (TP)' : '',
    'O Public administration and defence; compulsory social security, 2015 (TP)' : '',
    'P Education, 2015 (TP)' : '',
    'Q Human health and social work activities, 2015 (TP)' : '',
    'R Arts, entertainment and recreation, 2015 (TP)' : '',
    'S Other service activities, 2015 (TP)' : '',
    'T Activities of households as employers; undifferentiated goods- and services-producing activities of households for own use, 2015 (TP)' : '',
    'U Activities of extraterritorial organisations and bodies, 2015 (TP)' : '',
    'X Industry unknown, 2015 (TP)' : '',
    'Inhabitants, 2015 (PT)' : 'n_inhabitants_2015',
    'Labour force, 2015 (PT)' : 'n_labour_force_2015',
    'Employed, 2015 (PT)' : 'n_employed_2015',
    'Unemployed, 2015 (PT)' : 'n_unemployed_2015',
    'Persons outside the labour force, 2015 (PT)' : 'n_nonlabour_2015',
    'Children aged 0 to 14, 2015 (PT)' : '',
    'Students, 2015 (PT)' : '',
    'Pensioners, 2015 (PT)' : '',
    'Others, 2015 (PT)' : '',    
}



def aggregate_paavo(paavo_path=PAAVO_PATH):

    paavo_df = pd.read_csv(paavo_path, sep=';', na_values=['.', '..'])

    # Update the DF column names
    col_ids = [k if v == '' else v for k, v in column_map.items()]
    paavo_df.columns = col_ids

    # Leave out combined statistics of Finland
    paavo_df = paavo_df.iloc[1:]

    paavo_df['postal_code'] = paavo_df.iloc[:,0].str[:5].astype('str')
    paavo_df.set_index('postal_code', inplace=True)

    # Safe rows - No NaNs
    not_isnan_ix = np.logical_not(np.logical_or(
        np.isnan(paavo_df['n_households_2015']),
        np.isnan(paavo_df['n_households_highest_income_2015'])))

    paavo_df.loc[not_isnan_ix, 'n_households_highest_income_2015_pc'] = paavo_df.loc[not_isnan_ix, 'n_households_highest_income_2015'].astype(int)/paavo_df.loc[not_isnan_ix, 'n_households_2015'].astype(int)

    not_isnan_ix = np.logical_not(np.logical_or(
        np.isnan(paavo_df['n_households_2015']),
        np.isnan(paavo_df['n_inhabitants_highest_income_2015'])))

    paavo_df.loc[not_isnan_ix, 'n_inhabitants_highest_income_2015_pc'] = paavo_df.loc[not_isnan_ix, 'n_inhabitants_highest_income_2015'].astype(int)/paavo_df.loc[not_isnan_ix, 'n_inhabitants_2015'].astype(int)

    # Add a postal region (contains multiple postal areas) and region-ix columns (for stan mapping)
    paavo_df['postal_region'] = paavo_df.index.str[:2]
    postal_region_codes_df = pd.DataFrame(paavo_df['postal_region'].unique())
    postal_region_codes_df['postal_region_ix'] = range(1, len(postal_region_codes_df) + 1)
    region_ix = dict(postal_region_codes_df.values.tolist())
    postal_region_ix = [region_ix[region_code] for region_code in paavo_df['postal_region']]
    paavo_df['postal_region_ix'] = postal_region_ix         

    with open(op.join(op.dirname(__file__), '..', '..', 'paavodata_cleaned_df.pkl'), 'wb') as f:
        pickle.dump(paavo_df.sort_values(by='postal_code', ascending=True), f)
    
    return paavo_df


if __name__ == "__main__":
    aggregate_paavo()