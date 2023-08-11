
from typing import List, Literal
import argparse
import pickle
from torch_geometric.data import HeteroData, Data, Dataset
import warnings
import copy, os
import numpy as np
import random
from collections import defaultdict, Counter
import rdkit
import torch
import hashlib

from Bio.Data.IUPACData import protein_letters_3to1
from esm import pretrained
import Bio
import Bio.PDB
from collections import Counter
from nox.utils.protein_utils import (
    read_structure_file,
    filter_resolution,
    build_graph,
    compute_graph_edges,
    precompute_node_embeddings,
    compute_node_embedding,
    get_sequences,
)

from rxn.chemutils.smiles_randomization import randomize_smiles_rotated
from nox.utils.registry import register_object
from nox.datasets.abstract import AbstractDataset
from nox.utils.smiles import get_rdkit_feature, remove_atom_maps, assign_dummy_atom_maps
from nox.utils.pyg import from_smiles, from_mapped_smiles
from tdc.single_pred import ADME

@register_object("tdc_adme", "dataset")
class ADMEDataset(AbstractDataset):
    def load_dataset(self, args):
        data = ADME(name = args.tdc_dataset, path=args.dataset_file_path)
        self.metadata = data.get_split(
            method=args.split_type, 
            seed=args.split_seed, 
            frac=args.split_probs)
        self.tdc_dataset = args.tdc_dataset
    
    def create_dataset(self, split_group: Literal["train", "dev", "test"]) -> List[dict]:
        split = "valid" if split_group=="dev" else split_group
        dataset = self.metadata[split].to_dict('records')
        for i in range(len(dataset)):
            sample = dataset[i]
            sample['y'] = sample['Y']
        return dataset
    
    @property
    def protein_metadata(self):
        return {
            "CYP2C19": {
                "uniprot": "P33261",
                "sequence": "MDPFVVLVLCLSCLLLLSIWRQSSGRGKLPPGPTPLPVIGNILQIDIKDVSKSLTNLSKIYGPVFTLYFGLERMVVLHGYEVVKEALIDLGEEFSGRGHFPLAERANRGFGIVFSNGKRWKEIRRFSLMTLRNFGMGKRSIEDRVQEEARCLVEELRKTKASPCDPTFILGCAPCNVICSIIFQKRFDYKDQQFLNLMEKLNENIRIVSTPWIQICNNFPTIIDYFPGTHNKLLKNLAFMESDILEKVKEHQESMDINNPRDFIDCFLIKMEKEKQNQQSEFTIENLVITAADLLGAGTETTSTTLRYALLLLLKHPEVTAKVQEEIERVIGRNRSPCMQDRGHMPYTDAVVHEVQRYIDLIPTSLPHAVTCDVKFRNYLIPKGTTILTSLTSVLHDNKEFPNPEMFDPRHFLDEGGNFKKSNYFMPFSAGKRICVGEGLARMELFLFLTFILQNFNLKSLIDPKDLDTTPVVNGFASVPPFYQLCFIPV",
                "co_reactants": ["O=O", "Cc1cc2Nc3c([nH]c(=O)[nH]c3=O)N(C[C@H](O)[C@H](O)[C@H](O)COP([O-])([O-])=O)c2cc1C", "[H+]", "[H]O[H]"]  # "C12=NC([N-]C(C1=NC=3C(N2C[C@@H]([C@@H]([C@@H](COP(=O)([O-])[O-])O)O)O)=CC(=C(C3)C)C)=O)=O"
            },
            "CYP2D6": {
                "uniprot": "P10635",
                "sequence": "MGLEALVPLAVIVAIFLLLVDLMHRRQRWAARYPPGPLPLPGLGNLLHVDFQNTPYCFDQLRRRFGDVFSLQLAWTPVVVLNGLAAVREALVTHGEDTADRPPVPITQILGFGPRSQGVFLARYGPAWREQRRFSVSTLRNLGLGKKSLEQWVTEEAACLCAAFANHSGRPFRPNGLLDKAVSNVIASLTCGRRFEYDDPRFLRLLDLAQEGLKEESGFLREVLNAVPVLLHIPALAGKVLRFQKAFLTQLDELLTEHRMTWDPAQPPRDLTEAFLAEMEKAKGNPESSFNDENLRIVVADLFSAGMVTTSTTLAWGLLLMILHPDVQRRVQQEIDDVIGQVRRPEMGDQAHMPYTTAVIHEVQRFGDIVPLGVTHMTSRDIEVQGFRIPKGTTLITNLSSVLKDEAVWEKPFRFHPEHFLDAQGHFVKPEAFLPFSAGRRACLGEPLARMELFLFFTSLLQHFSFSVPTGQPRPSHHGVFAFLVSPSPYELCAVPR",
                "co_reactants": ["O=O", "Cc1cc2Nc3c([nH]c(=O)[nH]c3=O)N(C[C@H](O)[C@H](O)[C@H](O)COP([O-])([O-])=O)c2cc1C", "[H+]", "[H]O[H]"]  # "C12=NC([N-]C(C1=NC=3C(N2C[C@@H]([C@@H]([C@@H](COP(=O)([O-])[O-])O)O)O)=CC(=C(C3)C)C)=O)=O"
            },
            "CYP3A4": {
                "uniprot": "P08684",
                "sequence": "MALIPDLAMETWLLLAVSLVLLYLYGTHSHGLFKKLGIPGPTPLPFLGNILSYHKGFCMFDMECHKKYGKVWGFYDGQQPVLAITDPDMIKTVLVKECYSVFTNRRPFGPVGFMKSAISIAEDEEWKRLRSLLSPTFTSGKLKEMVPIIAQYGDVLVRNLRREAETGKPVTLKDVFGAYSMDVITSTSFGVNIDSLNNPQDPFVENTKKLLRFDFLDPFFLSITVFPFLIPILEVLNICVFPREVTNFLRKSVKRMKESRLEDTQKHRVDFLQLMIDSQNSKETESHKALSDLELVAQSIIFIFAGYETTSSVLSFIMYELATHPDVQQKLQEEIDAVLPNKAPPTYDTVLQMEYLDMVVNETLRLFPIAMRLERVCKKDVEINGMFIPKGVVVMIPSYALHRDPKYWTEPEKFLPERFSKKNKDNIDPYIYTPFGSGPRNCIGMRFALMNMKLALIRVLQNFSFKPCKETQIPLKLSLGGLLQPEKPVVLKVESRDGTVSGA",
                "co_reactants": ["O=O", "Cc1cc2Nc3c([nH]c(=O)[nH]c3=O)N(C[C@H](O)[C@H](O)[C@H](O)COP([O-])([O-])=O)c2cc1C", "[H+]", "[H]O[H]"]  # "C12=NC([N-]C(C1=NC=3C(N2C[C@@H]([C@@H]([C@@H](COP(=O)([O-])[O-])O)O)O)=CC(=C(C3)C)C)=O)=O"
            },
            "CYP1A2": {
                "uniprot": "P05177",
                "sequence": "MALSQSVPFSATELLLASAIFCLVFWVLKGLRPRVPKGLKSPPEPWGWPLLGHVLTLGKNPHLALSRMSQRYGDVLQIRIGSTPVLVLSRLDTIRQALVRQGDDFKGRPDLYTSTLITDGQSLTFSTDSGPVWAARRRLAQNALNTFSIASDPASSSSCYLEEHVSKEAKALISRLQELMAGPGHFDPYNQVVVSVANVIGAMCFGQHFPESSDEMLSLVKNTHEFVETASSGNPLDFFPILRYLPNPALQRFKAFNQRFLWFLQKTVQEHYQDFDKNSVRDITGALFKHSKKGPRASGNLIPQEKIVNLVNDIFGAGFDTVTTAISWSLMYLVTKPEIQRKIQKELDTVIGRERRPRLSDRPQLPYLEAFILETFRHSSFLPFTIPHSTTRDTTLNGFYIPKKCCVFVNQWQVNHDPELWEDPSEFRPERFLTADGTAINKPLSEKMMLFGMGKRRCIGEVLAKWEIFLFLAILLQQLEFSVPPGVKVDLTPIYGLTMKHARCEHVQARLRFSIN",
                "co_reactants": ["O=O", "Cc1cc2Nc3c([nH]c(=O)[nH]c3=O)N(C[C@H](O)[C@H](O)[C@H](O)COP([O-])([O-])=O)c2cc1C", "[H+]", "[H]O[H]"]  # "C12=NC([N-]C(C1=NC=3C(N2C[C@@H]([C@@H]([C@@H](COP(=O)([O-])[O-])O)O)O)=CC(=C(C3)C)C)=O)=O"
            },
            "CYP2C9": {
                "uniprot": "P11712",
                "sequence": "MDSLVVLVLCLSCLLLLSLWRQSSGRGKLPPGPTPLPVIGNILQIGIKDISKSLTNLSKVYGPVFTLYFGLKPIVVLHGYEAVKEALIDLGEEFSGRGIFPLAERANRGFGIVFSNGKKWKEIRRFSLMTLRNFGMGKRSIEDRVQEEARCLVEELRKTKASPCDPTFILGCAPCNVICSIIFHKRFDYKDQQFLNLMEKLNENIKILSSPWIQICNNFSPIIDYFPGTHNKLLKNVAFMKSYILEKVKEHQESMDMNNPQDFIDCFLMKMEKEKHNQPSEFTIESLENTAVDLFGAGTETTSTTLRYALLLLLKHPEVTAKVQEEIERVIGRNRSPCMQDRSHMPYTDAVVHEVQRYIDLLPTSLPHAVTCDIKFRNYLIPKGTTILISLTSVLHDNKEFPNPEMFDPHHFLDEGGNFKKSKYFMPFSAGKRICVGEALAGMELFLFLTSILQNFNLKSLVDPKNLDTTPVVNGFASVPPFYQLCFIPV",
                "co_reactants": ["O=O", "Cc1cc2Nc3c([nH]c(=O)[nH]c3=O)N(C[C@H](O)[C@H](O)[C@H](O)COP([O-])([O-])=O)c2cc1C", "[H+]", "[H]O[H]"]  # "C12=NC([N-]C(C1=NC=3C(N2C[C@@H]([C@@H]([C@@H](COP(=O)([O-])[O-])O)O)O)=CC(=C(C3)C)C)=O)=O"
            },    
        }
        
    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args
        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        AbstractDataset.add_args(parser)
        parser.add_argument(
            "--tdc_dataset",
            type=str,
            default=None,
            choices=["CYP2C19_Veith", "CYP2D6_Veith",  "CYP3A4_Veith",  "CYP1A2_Veith", "CYP2C9_Veith", "CYP2C9_Substrate_CarbonMangels", "CYP2D6_Substrate_CarbonMangels",  "CYP3A4_Substrate_CarbonMangels"],
            help="name of TDC dataset",
        )

    @staticmethod
    def set_args(args):
        args.dataset_file_path = "/Mounts/rbg-storage1/datasets/Metabo/TDC"


    def __getitem__(self, index):
        sample = self.dataset[index]
        sample_id = sample["Drug_ID"]
        try:
            # get protein data
            protein = self.tdc_dataset.split("_")[0]
            protein_meta = self.protein_metadata[protein]
            sequence = protein_meta["sequence"]
            uniprot_id = protein_meta['uniprot']
            co_reactants = protein_meta['co_reactants']
            
            drug = sample["Drug"]
            reactants = [drug] + co_reactants
            y = sample["Y"]
            reaction = ".".join(reactants)
            
            reactants = assign_dummy_atom_maps(reaction)
            reactants, atom_map2new_index = from_mapped_smiles(reactants, encode_no_edge=True)
            reactants.bond_changes = []
            
            item = {
                "reaction": reaction,
                "reactants": reactants,
                "sequence": sequence,
                "protein_id": uniprot_id,
                "sample_id": sample_id,
                "y": y
            }
            return item

        except Exception as e:
            warnings.warn(f"Could not load sample: {sample_id}")

@register_object("tdc_adme_substrates", "dataset")
class ADMESubstratesDataset(ADMEDataset):
    def __init__(self, args, split_group):
        super(ADMESubstratesDataset, ADMESubstratesDataset).__init__(self, args, split_group)
        esm_dir = "/Mounts/rbg-storage1/snapshots/metabolomics/esm2/checkpoints/esm2_t33_650M_UR50D.pt"
        self.esm_dir = esm_dir
        model, alphabet = pretrained.load_model_and_alphabet(esm_dir)
        self.esm_model = model
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()


    def __getitem__(self, index):
        sample = self.dataset[index]
        sample_id = sample["Drug_ID"]
        try:
            # get protein data
            protein = self.tdc_dataset.split("_")[0]
            protein_meta = self.protein_metadata[protein]
            sequence = protein_meta["sequence"]
            uniprot_id = protein_meta['uniprot']
            co_reactants = protein_meta['co_reactants']
            
            drug = sample["Drug"]
            reactants = [drug] + co_reactants
            y = sample["Y"]
            reaction = ".".join(reactants)
            
            # reactants = assign_dummy_atom_maps(reaction)
            # reactants, atom_map2new_index = from_mapped_smiles(reactants, encode_no_edge=True)
            # reactants.bond_changes = []

            item = {
                "reaction": reaction,
                "smiles": drug,
                # "reactants": reactants,
                "sequence": sequence,
                "protein_id": uniprot_id,
                "uniprot_id": uniprot_id,
                "sample_id": sample_id,
                "y": y
            }
            
            if self.args.use_protein_graphs:
                # load the protein graph
                graph_path = os.path.join(self.args.protein_graphs_dir, "processed", f"{item['uniprot_id']}_graph.pt")
                data = torch.load(graph_path) if os.path.exists(graph_path) else None
                if data is None:
                    structure_path = os.path.join(self.args.protein_structures_dir, f"AF-{item['uniprot_id']}-F1-model_v4.cif")
                    assert os.path.exists(structure_path), f"Structure path {graph_path} does not exist"
                    print(f"Structure path does exist, but graph path does not exist {graph_path}")
                    data = self.create_protein_graph(item)
                    torch.save(data, graph_path)

                data = self.add_additional_data_to_graph(data, item)
                if hasattr(data, "x") and not hasattr(data["receptor"], "x"):
                    data["receptor"].x = data.x

                keep_keys = {"receptor", "mol_data", "sequence", "protein_id", "uniprot_id", "sample_id", "smiles", "y", ('receptor', 'contact', 'receptor')}
                
                data_keys = data.to_dict().keys()
                for d_key in data_keys:
                    if not d_key in keep_keys:
                        delattr(data, d_key)

                coors = data["receptor"].pos
                feats = data["receptor"].x
                edge_index = data["receptor", "contact", "receptor"].edge_index
                assert coors.shape[0] == feats.shape[0], \
                    f"Number of nodes do not match between coors ({coors.shape[0]}) and feats ({feats.shape[0]})"

                assert max(edge_index[0]) < coors.shape[0] and max(edge_index[1]) < coors.shape[0], \
                    "Edge index contains node indices not present in coors"

                return data
            else: # just the substrate, with the protein sequence in the Data object
                reactant = from_smiles(item["smiles"])
                for key in item.keys():
                    reactant[key] = item[key]
                reactant["y"] = item["y"]
                return reactant


        except Exception as e:
            print(f"Getitem: Could not load sample: {sample_id} due to {e}")
            import traceback
            print(traceback.format_exc())
            
    def create_protein_graph(self, sample):
        try:
            raw_path = os.path.join(self.args.protein_structures_dir, f"AF-{sample['uniprot_id']}-F1-model_v4.cif")
            sample_id = sample["sample_id"]
            protein_parser = Bio.PDB.MMCIFParser()
            protein_resolution = "residue"
            graph_edge_args = {"knn_size": 10}
            center_protein = True
            esm_dir="/Mounts/rbg-storage1/snapshots/metabolomics/esm2/checkpoints/esm2_t33_650M_UR50D.pt"

            # parse pdb
            all_res, all_atom, all_pos = read_structure_file(
                protein_parser, raw_path, sample_id
            )
            # filter resolution of protein (backbone, atomic, etc.)
            atom_names, seq, pos = filter_resolution(
                all_res,
                all_atom,
                all_pos,
                protein_resolution=protein_resolution,
            )
            # generate graph
            data = build_graph(atom_names, seq, pos, sample_id)
            # kNN graph
            data = compute_graph_edges(data, **graph_edge_args)
            if center_protein:
                center = data["receptor"].pos.mean(dim=0, keepdim=True)
                data["receptor"].pos = data["receptor"].pos - center
                data.center = center
            uniprot_id = sample["uniprot_id"]
            sequence = sample['sequence']
            data.structure_sequence = sample['sequence']
            node_embeddings_args = {"model": self.esm_model, "model_location": self.esm_dir, "alphabet": self.alphabet, "batch_converter": self.batch_converter}

            embedding_path = os.path.join(self.args.protein_graphs_dir, "precomputed_node_embeddings", f"{sample['uniprot_id']}.pt")
            if os.path.exists(embedding_path):
                node_embedding = torch.load(
                    sample["embedding_path"]
                )
            else:
                node_embedding = compute_node_embedding(
                    data, **node_embeddings_args
                )
            # Fix sequence length mismatches
            if len(data["receptor"].seq) != node_embedding.shape[0]:
                print("Computing seq embedding for mismatched seq length")
                protein_letters_3to1.update({k.upper(): v for k, v in protein_letters_3to1.items()})
                AA_seq = ""
                for char in seq:
                    AA_seq += protein_letters_3to1[char]
                # sequences = get_sequences(
                #     self.protein_parser,
                #     [sample["sample_id"]],
                #     [os.path.join(self.structures_dir, f"AF-{sample['uniprot_id']}-F1-model_v4.cif")],
                # )
                
                data.structure_sequence = AA_seq
                data["receptor"].x = compute_node_embedding(
                    data, **node_embeddings_args
                )
            else:
                data["receptor"].x = node_embedding
            
            if len(data["receptor"].seq) != node_embedding.shape[0]:
                return None
            
            return data

        except Exception as e:
            import pdb; pdb.set_trace()
            print(f"Create prot graph: Could not load sample {sample['uniprot_id']} because of the exception {e}")
            return None

    def add_additional_data_to_graph(self, data, sample):
        skipped_keys = set(["protein_path", "embedding_path"])
        for key in sample.keys():
            if not key in skipped_keys and key not in data.to_dict().keys():
                data[key] = sample[key]
        data['smiles'] = sample['smiles']
        data['y'] = sample['y']
        data["mol_data"] = from_smiles(sample["smiles"])
        return data

    @staticmethod   
    def add_args(parser) -> None:
        """Add class specific args"""
        super(ADMESubstratesDataset, ADMESubstratesDataset).add_args(parser)
        parser.add_argument(
            "--use_protein_graphs",
            action="store_true",
            default=False,
            help="whether to use and generate protein graphs",
        )
        parser.add_argument(
            "--protein_graphs_dir",
            type=str,
            default=None,
            help="directory to load protein graphs from",
        )
        parser.add_argument(
            "--protein_structures_dir",
            type=str,
            default=None,
            help="directory to load protein graphs from",
        )
