import os
import csv
import array
import base64
import xmltodict

import numpy as np

"""
출처
source from https://github.com/hewittwill/ECGXMLReader.git
"""

__author__ = "Will Hewitt"
__credits__ = ["Will Hewitt"]
__version__ = "1.0.0"
__maintainer__ = "Will Hewitt"
__email__ = "me@hewittwill.com"
__status__ = "Development"


class ECGXMLReader:
    """ Extract voltage data from a ECG XML file """

    def __init__(self, path, name, augmentLeads=False):
        try:
            with open(path, 'rb') as xml:
                self.ECG = xmltodict.parse(xml.read().decode('ISO-8859-1'))  # ISO-8859-1

            self.augmentLeads = augmentLeads
            self.path = path
            self.name = name

            self.PatientDemographics = self.ECG['RestingECG']['PatientDemographics']
            #             self.TestDemographics       = self.ECG['RestingECG']['TestDemographics']
            #             self.RestingECGMeasurements = self.ECG['RestingECG']['RestingECGMeasurements']
            self.Waveforms = self.ECG['RestingECG']['Waveform']

            self.LeadVoltages = self.makeLeadVoltages()

        except Exception as e:
            print(str(e))

    def makeLeadVoltages(self):

        num_leads = 0

        leads = {}

        if self.name[0] == '6':
            for lead in self.Waveforms['LeadData']:
                num_leads += 1

                lead_data = lead['WaveFormData']
                lead_b64 = base64.b64decode(lead_data)
                lead_vals = np.array(array.array('h', lead_b64))

                leads[lead['LeadID']] = lead_vals
        elif self.name[0] == '5':
            for lead in self.Waveforms[1]['LeadData']:  ##median ignore, only rhythm
                num_leads += 1

                lead_data = lead['WaveFormData']
                lead_b64 = base64.b64decode(lead_data)
                lead_vals = np.array(array.array('h', lead_b64))

                leads[lead['LeadID']] = lead_vals
        else:  # self.name[0] == '8'
            for lead in self.Waveforms['LeadData']:
                num_leads += 1

                lead_data = lead['WaveFormData']
                lead_b64 = base64.b64decode(lead_data)
                lead_vals = np.array(array.array('h', lead_b64))

                leads[lead['LeadID']] = lead_vals

        if num_leads == 8 and self.augmentLeads:
            leads['III'] = np.subtract(leads['II'], leads['I'])
            leads['aVR'] = np.add(leads['I'], leads['II']) * (-0.5)
            leads['aVL'] = np.subtract(leads['I'], 0.5 * leads['II'])
            leads['aVF'] = np.subtract(leads['II'], 0.5 * leads['I'])

        return leads

    def getLeadVoltages(self, LeadID):
        return self.LeadVoltages[LeadID]

    def getAllVoltages(self):
        return self.LeadVoltages
