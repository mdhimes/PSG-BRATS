"""
Utility functions and lists used by pypsgplus.

enumerated_product: uses itertools.product with enumerate-like index counting.

make_dir: handles creating directories.

tprint: prints a date-time stamp at the beginning of the requested print statement.

time_elapsed: calculates the time elapsed between two datetime objects.
"""

import sys, os
import datetime
import itertools
import numpy as np


molecules = ['He', 'H2O', 'CO2', 'O3', 'N2O', 'CO', 'CH4', 'O2', 'NO', 'SO2', 
             'NO2', 'NH3', 'HNO3', 'OH', 'HF' 'HCl', 'HBr', 'HI', 'ClO', 'OCS', 
             'H2CO', 'HOCl', 'N2', 'HCN', 'CH3Cl', 'H2O2', 'C2H2', 'C2H6', 
             'PH3', 'COF2', 'SF6', 'H2S', 'HCOOH', 'HO2', 'O', 'ClONO2', 'NO+', 
             'HOBr', 'C2H4', 'CH3OH', 'CH3Br', 'CH3CN', 'CF4', 'C4H2', 'HC3N', 
             'H2', 'CS', 'SO3', 'C2N2', 'COCl2', 'SO', 'CH3F', 'GeH4', 'CS2', 
             'CH3I', 'NF3', 'CH3D', 'C3H8', 'C3H4', 'HNC']

aerosols = ['5_Myhre_HNO3_273K_54p_HRI',
            '5_Norman_HNO3_54p_HRI',
            '5_Biermann_HNO3_253K_50p_HRI',
            '5_Remsberg_H2SO4_HNO3_75p_HRI',
            '5_Remsberg_H2SO4_HNO3_90p_HRI',
            '5_Remsberg_H2SO4_HNO3_68p_HRI',
            '5_Myhre_17pHNO3_25pH2SO4_58pH2O_HRI',
            '5_Niedziela_NAD_HRI',
            '5_Richwine_NAT_HRI',
            '5_Toon_PSC_betanat_HRI',
            '5_Toon_PSC_nad_HRI',
            '5_Toon_PSC_alphanat_HRI',
            '5_Toon_PSC_nam_HRI',
            '5_Toon_PSC_ice_163K_HRI',
            '5_Toon_PSC_anat_HRI',
            '5_Toon_PSC_anad_HRI',
            '5_Toon_PSC_anam_HRI',
            '5_AFCRL_Carbonaceous_HRI',
            '5_Ice_Martonchik_GSFC',
            '5_Hansen_GSFC',
            '5_White_GSFC',
            '5_Simple_GSFC',
            '5_AFCRL_Volcanic_dust_HRI',
            '5_AFCRL_Meteor_dust_HRI',
            '5_AFCRL_Dust_like_HRI',
            '5_Querry_clay_HRI',
            '5_Querry_illite_HRI',
            '5_Querry_kaolin_HRI',
            '5_Querry_mont_HRI',
            '5_Toon_granite_HRI',
            '5_Wagner_sahara_dust_HRI',
            '5_Begemann_Al2O3_HRI',
            '5_Henning_FeO_HRI',
            '5_Posch_CaTiO3_HRI',
            '5_Triaud_Fe203_HRI',
            '5_Jena_Fe2SiO4_HRI',
            '5_Fabian_Fe2SiO4_HRI',
            '5_Fabian_MgAl2O4_HRI',
            '5_Jager_Mg2SiO4_HRI',
            '5_Jager_MgSiO3_HRI',
            '5_Zeidler_TiO2_Rutile_HRI',
            '5_Zeidler_TiO2_Anatase_HRI',
            '5_Posch_TiO2_Brookite_HRI',
            '5_Querry_KCl_HRI',
            '5_Querry_ZnS_HRI',
            '4_CRISM_Wolff',
            '5_Sutherland_biomass_fire_HRI',
            '5_Magi_biomass_fire_HRI',
            '5_Stagg_C_flame_pyro_400C_HRI',
            '5_Stagg_C_flame_amorphous_400C_HRI',
            '5_Stagg_C_flame_propane_300C_HRI',
            '5_Stagg_C_flame_soot_300C_HRI',
            '5_Chang_flame_soot_HRI',
            '5_Querry_diesel_soot_HRI',
            '5_Alexander_brown_carbon_HRI',
            '5_Myhre_organic_acids_Oxalic_10p_HRI',
            '5_Myhre_organic_acids_Malonic_25p_HRI',
            '5_Myhre_organic_acids_Succinc_8p_HRI',
            '5_Myhre_organic_acids_Glutaric_25p_HRI',
            '5_Myhre_organic_acids_Pyruvic_5p_HRI',
            '5_Khare_Titan_Tholins_HRI',
            '5_Ramirez_Titan_aerosol_HRI',
            '5_AFCRL_Hematite_oray_HRI',
            '5_AFCRL_Hematite_eray_HRI',
            '5_Ice_Martonchik_GSFC',
            '5_Crystalline_GSFC',
            '5_AFCRL_Quartz_oray_HRI',
            '5_AFCRL_Quartz_eray_HRI',
            '5_AFCRL_Sand_oray_HRI',
            '5_AFCRL_Sand_eray_HRI',
            '5_AFCRL_Seasalt_HRI',
            '5_AFCRL_NaCl_HRI',
            '5_AFCRL_Ammonium_HRI',
            '5_AFCRL_H2SO4_215K_HRI',
            '5_AFCRL_H2SO4_300K_HRI',
            '5_Tisdale_H2SO4_60p_HRI',
            '5_Niedziela_H2SO4_220K_50p_HRI',
            '5_Myhre_H2SO4_298K_81p_HRI',
            '5_Biermann_H2SO4_253K_60p_HRI',
            '5_PalmerWilliams_H2SO4_75p_HRI',
            '5_Grainger_Volcanic_ash_HRI',
            '5_Henning_low_T_SiO2_HRI',
            '5_Zeidler_high_T_SiO2_HRI',
            '5_Downing_Williams_Water_HRI',
            '5_AFCRL_Water_HRI',
            '5_AFCRL_Water_soluble_HRI',
            '5_AFCRL_Ice_HRI',
            '5_Wagner_ice_238K_HRI',
            '5_Wagner_ice_252K_HRI',
            '5_Wagner_ice_258K_HRI',
            '5_Wagner_ice_269K_HRI',
            '5_Warren_ice_HRI',
            '5_Clapp_ice_130K_HRI',
            '5_Clapp_ice_140K_HRI',
            '5_Clapp_ice_150K_HRI',
            '5_Clapp_ice_160K_HRI',
            '5_Clapp_ice_170K_HRI',
            '5_Clapp_ice_180K_HRI',
            '5_Clapp_ice_190K_HRI',
            '5_Clapp_ice_200K_HRI',
            '4_CRISM_Wolff']

watm_config_keys = ['ATMOSPHERE-STRUCTURE', 'ATMOSPHERE-WEIGHT', 'ATMOSPHERE-ABUN', 'ATMOSPHERE-LAYERS-MOLECULES', 
                    'SURFACE-TEMPERATURE'] + ['ATMOSPHERE-LAYER-'+str(i+1) for i in range(300)]

kempton_keys = ['KEMPTON-PROFILE', 'KEMPTON-TFILL', 'KEMPTON-TYPE', 'KEMPTON-VAL']

tea_molecules = ['H', 'He', 'C', 'N', 'O', 'Na', 'K', 'H2', 'CO', 'CO2', 
                 'CH4', 'H2O', 'N2', 'NH3', 'HCN', 'C2H2', 'C2H4', 'C2H6', 
                 'COS', 'H2S', 'H2CO', 'O2', 'O3', 'NO', 'NO2', 'SO2', 
                 'OH', 'TiO', 'VO', 'MgH', 'PH3', 'HCl', 'SiH', 'SiO', 'HF']
tea_keys = ['TEA-MH', 'TEA-C2O']

# DO NOT change the order of these, without also fixing the logic in PT.py!
pt_keys  = ['PT-GAMMA', 'PT-T0', 'PT-LOGP0', # 'adiabat' model
            'PT-KAPPA', 'PT-GAMMA1', 'PT-GAMMA2', 'PT-ALPHA', 'PT-BETA', # 'line' model
            'PT-DT-BOT-32', 'PT-DT-32-10', 'PT-DT-10-0', 'PT-DT-0-1', 'PT-DT-1-01', 'PT-DT-01-001', 
            'PT-DT-001-TOP'] #'spt' model

algorithms = ['demc', 'dream', 'emcee', 'snooker', # MCMC
              'dnest4', 'dynesty', 'multinest', 'polychord', 'ultranest'] # Nested sampling


def enumerated_product(*args):
    """
    Helper function to use itertools.product with enumerate-like index counting
    """
    return list(zip(itertools.product(*(range(len(x)) for x in args)), itertools.product(*args)))


def make_dir(some_dir):
    """
    Handles creation of a directory.

    Inputs
    ------
    some_dir: string. Directory to be created.

    Outputs
    -------
    None. Creates `some_dir` if it does not already exist. 
    Raises an error if the directory cannt be created.
    """
    try:
      os.mkdir(some_dir)
    except OSError as e:
      if e.errno == 17: # Already exists
        pass
      else:
        print("Cannot create folder '{:s}'. {:s}.".format(some_dir, os.strerror(e.errno)))
        sys.exit()
    return


def tprint(*args, **kwargs):
    now = datetime.datetime.now()
    print(str(now)[:19], "--", *args, **kwargs)
    return now


def time_elapsed(time_start, time_end):
    deltaobj = (time_end - time_start)
    delta    = deltaobj.seconds + deltaobj.microseconds/1e6
    hours   = delta//60//60
    minutes = delta//60%60
    seconds = np.round(delta%60, 2)
    ret = []
    if hours:
        ret.append(str(hours) + " hrs")
    if minutes:
        ret.append(str(minutes) + " min")
    ret.append(str(seconds) + " sec")
    return " ".join(ret)


