#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os
# test_path = os.path.abspath(os.path.join(os.getcwd(),os.pardir))
# root_path = os.path.abspath(os.path.join(test_path,os.pardir))
# data_path = os.path.abspath(os.path.join(root_path,'nemesispy/data'))
# ktable_path = os.path.abspath(os.path.join(data_path,'ktables'))
# cia_path = os.path.abspath(os.path.join(data_path,'cia'))

ktable_path = "/gf3/planetary2/PGJI011_YANG_EXOPHASE/nemesispy2022/nemesispy/data/ktables"
lowres_file_paths = [
    'h2owasp43.kta',
    'co2wasp43.kta',
    'cowasp43.kta',
    'ch4wasp43.kta']
for ipath,path in enumerate(lowres_file_paths):
    lowres_file_paths[ipath] = os.path.join(ktable_path,path)

ariel_file_paths = [
    'H2O_Katy_ARIEL_test.kta',
    'CO2_Katy_ARIEL_test.kta',
    'CO_Katy_ARIEL_test.kta',
    'CH4_Katy_ARIEL_test.kta']
for ipath,path in enumerate(ariel_file_paths):
    ariel_file_paths[ipath] = os.path.join(ktable_path,path)

hires_file_paths = [
    'H2O_Katy_ARIEL_test.kta',
    'CO2_Katy_ARIEL_test.kta',
    'CO_Katy_ARIEL_test.kta',
    'CH4_Katy_ARIEL_test.kta']
for ipath,path in enumerate(ariel_file_paths):
    ariel_file_paths[ipath] = os.path.join(ktable_path,path)

cia_file_path = "/gf3/planetary2/PGJI011_YANG_EXOPHASE/nemesispy2022/nemesispy/data/cia"
cia_file_path = os.path.join(cia_file_path,'exocia_hitran12_200-3800K.tab')
