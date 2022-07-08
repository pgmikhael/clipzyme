from xml.etree import cElementTree as ET
from collections import defaultdict
import json
import argparse
from tqdm import tqdm 
import re

HMDB_Metabolites = "/Mounts/rbg-storage1/datasets/Metabo/HMDB/hmdb_metabolites.xml"
HMDB_Proteins = "/Mounts/rbg-storage1/datasets/Metabo/HMDB/hmdb_proteins.xml"
HMDB_SPECTRA = "/Mounts/rbg-storage1/datasets/Metabo/HMDB/hmdb_all_spectra"


parser = argparse.ArgumentParser(description="HMDB Parser.")
parser.add_argument(
    "--save_metabolites_path",
    type=str,
    default="/Mounts/rbg-storage1/datasets/Metabo/HMDB/metabolites.json",
    help="path to metabolites metadata json file",
)
parser.add_argument(
    "--save_proteins_path",
    type=str,
    default="/Mounts/rbg-storage1/datasets/Metabo/HMDB/proteins.json",
    help="path to protein metadata json file",
)



def xml2dict(t):
    d = {}
    children = list(t)
    if children:
        dd = defaultdict(list)
        for dc in map(xml2dict, children):
            for k, v in dc.items():
                if (not k == "") and not ("reference" in k):
                    dd[k.split("}")[-1]].append(v)
        d = {
            t.tag.split("}")[-1]: {
                k.split("}")[-1]: v[0] if len(v) == 1 else v for k, v in dd.items()
            }
        }

    if t.text:
        text = t.text.strip()
        if (not text == "") and not ("reference" in text):
            d[t.tag] = text
    return d

def add_metabolite_spectra(mid, mdict, metabolite_spectra_files):
    hmdbid_to_spectrum = {}
    for f in metabolite_spectra_files:
        parsed = os.path.splitext(f)[0].split('_')
        typeid = parsed.index('spectrum')
        stype = '_'.join(parsed[1:typeid])
        
        if parsed[-1] == "predicted":
            sid = '{}.{}'.format(parsed[-2],stype)
            hmdbid_to_spectrum[sid] = (f, "predicted", stype, parsed[-2])
        elif parsed[-1] == "experimental":
            sid = '{}.{}'.format(parsed[-2],stype)
            hmdbid_to_spectrum[sid] = (f, "experimental", stype, parsed[-2])
        else:
            sid = '{}.{}'.format(parsed[-1],stype)
            hmdbid_to_spectrum[sid] = (f, None, stype,  parsed[-1])

    assert len(hmdbid_to_spectrum) == len(metabolite_spectra_files)
    
    if 'spectra' not in mdict:
        mdict['spectra'] = {'spectrum': []}

    if not isinstance(mdict['spectra']['spectrum'], list):
        mdict['spectra']['spectrum'] = [mdict['spectra']['spectrum']]

    for spect in mdict['spectra']['spectrum']:
        stype = spect['type'].split('Specdb::')[-1]
        stype = '_'.join(re.findall('[A-Z][^A-Z]*', stype)).lower()
        sid = '{}.{}'.format(spect['spectrum_id'], stype)
        spect['file'] = hmdbid_to_spectrum.get(sid, (None,None) )[0]
        spect['type'] = stype
        spect['experimental_or_predicted'] = hmdbid_to_spectrum.pop(sid, (None,None) )[1]

    for _, (fl, status, stype, spid) in hmdbid_to_spectrum.items():
        mdict['spectra']['spectrum'].append({
            'file': fl,
            'experimental_or_predicted': status,
            'spectrum_id':spid,
            'type': stype
            })
    
if __name__ == "__main__":

    args = parser.parse_args()

    spectra_files = os.listdir(HMDB_SPECTRA) 
    metabolite_spectra_files=defaultdict(list)
    for l in spectra_files:
        metabolite_spectra_files[l.split('_')[0]].append(l)

    metabolites_db = {}
    with tqdm() as tqdm_bar:
        for event, elem in ET.iterparse(HMDB_Metabolites, events=("start", "end")):

            if (elem.tag == "{http://www.hmdb.ca}metabolite") and event == "end":
                xml_data = ET.XML(ET.tostring(elem))
                d = xml2dict(xml_data)["metabolite"]
                for k in [
                    "taxonomy",
                    "description",
                    "ontology",
                    "normal_concentrations",
                    "abnormal_concentrations",
                    "synthesis_reference",
                    "general_references",
                ]:
                    if k in d:
                        del d[k]
                add_metabolite_spectra(d['accession'], d, metabolite_spectra_files[d['accession']])
                metabolites_db[d['accession']] = d
                tqdm_bar.update()

    json.dump(metabolites_db, open(args.save_metabolites_path, "w"))
    
    proteins_db = {}
    with tqdm() as tqdm_bar:
        for event, elem in ET.iterparse(HMDB_Metabolites, events=("start", "end")):
            if (elem.tag == "{http://www.hmdb.ca}protein") and event == "end":
                xml_data = ET.XML(ET.tostring(elem))
                d = xml2dict(xml_data)["protein"]
                proteins_db[d['protein_accession']] = d
                tqdm_bar.update()

    json.dump(proteins_db, open(args.save_proteins_path, "w"))
