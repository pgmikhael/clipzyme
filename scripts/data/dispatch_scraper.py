import subprocess
import os
import multiprocessing
from argparse import ArgumentParser
import json
from csv import DictWriter
import time

parser = ArgumentParser(
    description="Script to run bigg_scraper for organisms in parallel."
)
parser.add_argument(
    "--organisms",
    type=str,
    nargs="*",
    default=[
        "iAB_RBC_283",
        "iAF1260",
        "iAF1260b",
        "iAF692",
        "iAF987",
        "iAM_Pb448",
        "iAM_Pc455",
        "iAM_Pf480",
        "iAM_Pk459",
        "iAM_Pv461",
        "iAPECO1_1312",
        "iAT_PLT_636",
        "iB21_1397",
        "iBWG_1329",
        "ic_1306",
        "iCHOv1",
        "iCHOv1_DG44",
        "iCN718",
        "iCN900",
        "iE2348C_1286",
        "iEC042_1314",
        "iEC1344_C",
        "iEC1349_Crooks",
        "iEC1356_Bl21DE3",
        "iEC1364_W",
        "iEC1368_DH5a",
        "iEC1372_W3110",
        "iEC55989_1330",
        "iECABU_c1320",
        "iECB_1328",
        "iECBD_1354",
        "iECD_1391",
        "iECDH10B_1368",
        "iEcDH1_1363",
        "iECDH1ME8569_1439",
        "iEcE24377_1341",
        "iECED1_1282",
        "iECH74115_1262",
        "iEcHS_1320",
        "iECIAI1_1343",
        "iECIAI39_1322",
        "iECNA114_1301",
        "iECO103_1326",
        "iECO111_1330",
        "iECO26_1355",
        "iECOK1_1307",
        "iEcolC_1368",
        "iECP_1309",
        "iECs_1301",
        "iECS88_1305",
        "iECSE_1348",
        "iECSF_1327",
        "iEcSMS35_1347",
        "iECSP_1301",
        "iECUMN_1333",
        "iECW_1372",
        "iEK1008",
        "iEKO11_1354",
        "iETEC_1333",
        "iG2583_1286",
        "iHN637",
        "iIS312",
        "iIS312_Amastigote",
        "iIS312_Epimastigote",
        "iIS312_Trypomastigote",
        "iIT341",
        "iJB785",
        "iJN1463",
        "iJN678",
        "iJN746",
        "iJO1366",
        "iJR904",
        "iLB1027_lipid",
        "iLF82_1304",
        "iLJ478",
        "iML1515",
        "iMM1415",
        "iMM904",
        "iND750",
        "iNF517",
        "iNJ661",
        "iNRG857_1313",
        "iPC815",
        "iRC1080",
        "iS_1188",
        "iSB619",
        "iSbBS512_1146",
        "iSBO_1134",
        "iSDY_1059",
        "iSF_1195",
        "iSFV_1184",
        "iSFxv_1172",
        "iSSON_1240",
        "iSynCJ816",
        "iUMN146_1321",
        "iUMNK88_1353",
        "iUTI89_1310",
        "iWFL_1372",
        "iY75_1357",
        "iYL1228",
        "iYO844",
        "iYS1720",
        "iYS854",
        "iZ_1308",
        "STM_v1_0",
    ],
    help="organism name in bigg database.",
)
parser.add_argument("--cpus", type=int, default=2, help="Number of cpus.")
parser.add_argument(
    "--save_dir",
    type=str,
    default="/Mounts/rbg-storage1/datasets/Metabo/datasets/",
    help="directory to save the dataset",
)


def launch_job(flag_string, filepath, logpath):
    shell_cmd = "python -u scripts/data/bigg_scraper.py {} > {} 2>&1".format(
        flag_string, logpath
    )
    print("Launched command: {}".format(shell_cmd))
    if not os.path.exists(filepath):
        subprocess.call(shell_cmd, shell=True)

    return filepath, logpath


def worker(job_queue, done_queue):
    """
    Worker thread for each gpu. Consumes all jobs and pushes results to done_queue.
    :gpu - gpu this worker can access.
    :job_queue - queue of available jobs.
    :done_queue - queue where to push results.
    """
    while not job_queue.empty():
        flag_string, datapath, logpath, organism = job_queue.get()
        if flag_string is None:
            return
        done_queue.put((launch_job(flag_string, datapath, logpath), organism))


if __name__ == "__main__":
    args = parser.parse_args()
    job_list = [
        (
            f"--organism_name {organism} --save_dir {args.save_dir}",
            os.path.join(args.save_dir, f"{organism}_dataset.json"),
            os.path.join(args.save_dir, f"logs/{organism}_dataset.log"),
            organism,
        )
        for organism in args.organisms
    ]

    job_queue = multiprocessing.Queue()
    done_queue = multiprocessing.Queue()

    for job in job_list:
        job_queue.put(job)

    for cpu in range(args.cpus):
        multiprocessing.Process(target=worker, args=(job_queue, done_queue)).start()

    for i in range(len(job_list)):
        (filepath, logpath), organism = done_queue.get()
        summary_result = {
            "organism": organism,
            "status": None,
            "log_file": logpath,
            "date": time.strftime("%m-%d-%Y %H:%M:%S", time.localtime()),
        }
        try:
            result_dict = json.load(open(filepath, "r"))
            print(
                "({}/{}) \t SUCCESS: CREATED {}".format(i + 1, len(job_list), organism)
            )
            summary_result["status"] = "success"
        except Exception as e:
            print(
                "({}/{}) \t FAILED TO CREATE: {}".format(i + 1, len(job_list), organism)
            )
            summary_result["status"] = "failed"

        with open(
            os.path.join(args.save_dir, "dataset_creation_summary.csv"), "a", newline=""
        ) as f_object:
            dictwriter_object = DictWriter(
                f_object, fieldnames=["organism", "status", "log_file", "date"]
            )
            dictwriter_object.writerow(summary_result)
