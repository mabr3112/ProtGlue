'''Full Python Script of the StickyProt Computational pipeline for design of protein binders.'''
## system
import sys
import random
import argparse
import os
import logging

# dependencies
import numpy as np
import protflow
from protflow.poses import Poses
from protflow.jobstarters import SbatchArrayJobstarter, LocalJobStarter
from protflow.tools.rfdiffusion import RFdiffusion
from protflow.tools.ligandmpnn import LigandMPNN
from protflow.tools.esmfold import ESMFold
from protflow.metrics.rmsd import MotifRMSD
from protflow.utils.metrics import calc_rog_of_pdb
import protflow.utils.plotting as plots
from protflow.tools.residue_selectors import ChainSelector, TrueSelector
from protflow.tools.colabfold import Colabfold, calculate_poses_interaction_pae
from protflow.tools.protein_edits import SequenceRemover, ChainAdder
from protflow.utils.metrics import calc_interchain_contacts_pdb, residue_contacts
from protflow.config import AUXILIARY_RUNNER_SCRIPTS_DIR as scripts_dir
from protflow.tools.rosetta import Rosetta
from protflow.utils.biopython_tools import load_structure_from_pdbfile, get_sequence_from_pose

# here we define miscellaneous functions
def random_numbers() -> str:
    '''Guess what it does.'''
    return "".join([str(x) for x in random.sample(range(1,10), 9)])

def compile_dimer_msa_str(b_seq, t_seq, target_msa_sequences) -> str:
    '''Creates fake msa_str for alphafold prediction. very custom, not to be used by anyone.'''
    # compile gap strings for target and binder
    b_len = len(b_seq)
    t_len = len(t_seq)
    b_gap = "-"*b_len
    t_gap = '-'*t_len

    # write msa-str for target
    msa_target = "\n".join([f">target_{str(i).zfill(4)}\t{random_numbers()}\t{random_numbers()}\n{b_gap}{seq}" for i, seq in enumerate(target_msa_sequences, start=1)])

    # write str for colabfold_batch input fasta.
    msa_str = f"""#{b_len},{t_len},{t_len}\t1,1,1
>101\t102\t103
{b_seq}{t_seq}{t_seq}
>101
{b_seq}{t_gap}{t_gap}
>102
{b_gap}{t_seq}{t_gap}
{msa_target}
>103
{b_gap}{t_gap}{t_seq}
{msa_target}
"""
    return msa_str

def compile_msa_str(b_seq: str, t_seq: str, target_msa_sequences: list[str], paired_msa_sequences: list[str]) -> str:
    '''Compiles a3m file formatted msa string for binder-target prediction using target_msa_sequences and joint_msa_sequences.'''
    # compile gap strings for target and binder
    b_len = len(b_seq)
    t_len = len(t_seq)
    b_gap = "-"*b_len
    t_gap = "-"*t_len

    # write joint msa-str
    msa_joint = "\n".join(f">joint_{str(i).zfill(4)}\t{random_numbers()}\n{seq.replace(':','')}" for i, seq in enumerate(paired_msa_sequences, start=1))

    # write target msa-str
    msa_target = "\n".join(f">target_{str(i).zfill(4)}\t{random_numbers()}\n{b_gap}{seq}" for i, seq in enumerate(target_msa_sequences, start=1))

    # compile string for joint MSA:
    msa_str = f"""#{b_len},{t_len}\t1,1
>101\t102
{b_seq}{t_seq}
{msa_joint}
>101
{b_seq}{t_gap}
>102
{b_gap}{t_seq}
{msa_target}
"""
    return msa_str

## This defines what is done when the script is started as a script
def main(args):
    '''Main function that processes poses using LigandMPNN.'''
    # set logging
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        filename=f'{args.output_dir}/rfdiffusion_protflow_log.txt',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # setup jobstarters
    if args.jobstarter == "slurm_gpu":
        optional_gpu_jobstarter = SbatchArrayJobstarter(max_cores=args.max_gpus, gpus=1)
        gpu_jobstarter = SbatchArrayJobstarter(max_cores=args.max_gpus, gpus=1)
        cpu_jobstarter = SbatchArrayJobstarter(max_cores=args.max_cpus)
    elif args.jobstarter == "slurm_cpu":
        optional_gpu_jobstarter = SbatchArrayJobstarter(max_cores=args.max_cpus)
        gpu_jobstarter = SbatchArrayJobstarter(max_cores=args.max_gpus, gpus=1)
        cpu_jobstarter = SbatchArrayJobstarter(max_cores=args.max_cpus)
    elif args.jobstarter == "local":
        optional_gpu_jobstarter = LocalJobStarter(max_cores=args.max_gpus)
        gpu_jobstarter = LocalJobStarter(max_cores=args.max_gpus)
        cpu_jobstarter = LocalJobStarter(max_cores=args.max_cpus)
    else:
        raise ValueError(f"Either {{slurm_gpu, slurm_cpu, local}} are allowed as options for --jobstarter parameter.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Create instances of necessary classes
    ligandmpnn_runner = LigandMPNN(jobstarter=optional_gpu_jobstarter)
    rfdiffusion_runner = RFdiffusion(jobstarter=optional_gpu_jobstarter)
    esmfold_runner = ESMFold(jobstarter=gpu_jobstarter)
    colabfold_runner = Colabfold(jobstarter=optional_gpu_jobstarter)
    motif_bb_rmsd = MotifRMSD(ref_col="rfdiff_1_location", jobstarter=cpu_jobstarter, atoms=["N", "CA", "C", "O"])
    rosetta=Rosetta(jobstarter=cpu_jobstarter)

    # setup chain selectors
    chain_selector = ChainSelector()
    true_selector = TrueSelector()
    chain_adder = ChainAdder(jobstarter=cpu_jobstarter)
    sequence_remover = SequenceRemover(jobstarter=cpu_jobstarter)

    # Load poses from the input directory
    poses = Poses(poses=args.target, work_dir=args.output_dir)
    target_seq = get_sequence_from_pose(load_structure_from_pdbfile(poses.poses_list()[0]))
    if len(target_seq.split(":")) > 1:
        raise ValueError(f"Target must be single-chain. Otherwise this pipeline will fail!")

    # setup results directory
    results_dir = f"{poses.work_dir}/results/"
    os.makedirs(results_dir, exist_ok=True)

    # define target contig.
    if args.target_contig:
        target_residues = protflow.residues.from_contig(args.target_contig)
        target_length = len(target_residues)
        target_contig = args.target_contig
    else:
        true_selector.select("target_residues", poses)
        target_residues = poses.df["target_residues"].values[0].to_rfdiffusion_contig()
        target_length = len(target_residues)
        target_contig = target_residues

    diff_opts = f"diffuser.T=50 'contigmap.contigs=[{target_contig}/0 {args.binder_length}-{args.binder_length}]' 'ppi.hotspot_res=[{args.hotspot_residues}]' potentials.guiding_potentials=[\\'type:custom_binder_potential,binderlen:{args.binder_length},contacts_weight:{args.contacts_weight},rog_weight:{args.rog_weight}\\']"

    # Diffuse binders to target
    logging.info(f"Starting diffusion of {args.num_diff} binders on {rfdiffusion_runner.jobstarter.max_cores} cores.")
    multiplex_poses = args.multiplex_rfdiffusion or None
    poses = rfdiffusion_runner.run(
        poses=poses,
        prefix="rfdiff_1",
        num_diffusions=args.num_diff,
        options=diff_opts,
        multiplex_poses=multiplex_poses
    )

    # calculate rfdiffusion binder stats
    poses.df["rfdiffusion_rog"] = [calc_rog_of_pdb(pose, chain="A") for pose in poses.poses_list()]
    poses.df["rfdiffusion_binder_contacts"] = [calc_interchain_contacts_pdb(pose, chains=["A", "B"]) for pose in poses.poses_list()]
    poses.df["rfdiffusion_hotspot_contacts"] = [np.sum([residue_contacts(pose, max_distance=10, target_chain=residue[0], partner_chain="A", target_resnum=int(residue[1:])+int(args.binder_length), min_distance = 4) for residue in args.hotspot_residues.split(",")]) for pose in poses.poses_list()]

    # plot rfdiffusion results
    plots.violinplot_multiple_cols(
        dataframe=poses.df,
        cols = ["rfdiff_1_plddt", "rfdiffusion_rog", "rfdiffusion_binder_contacts", "rfdiffusion_hotspot_contacts"],
        y_labels = ["pLDDT", "ROG [\u00C5]", "# contacting Ca", "# hotspot contacts"],
        titles = ["Rfdiffusion pLDDT", "ROG", "Contacts", "Hotspot Contacts"],
        out_path = f"{results_dir}/rfdiffusion_stats.png"
    )

    # remove unusable diffusion outputs
    poses.filter_poses_by_value("rfdiff_1_plddt", args.rfdiff_plddt_cutoff, ">", prefix="rfdiff_plddt_filter", plot=True)
    poses.filter_poses_by_value("rfdiffusion_rog", args.rog_cutoff, "<", prefix="rfdiffusion_rog_filter", plot=True)
    poses.filter_poses_by_value("rfdiffusion_binder_contacts", args.contacts_cutoff, ">", prefix="rfdiffusion_binder_contacts_filter", plot=True)
    poses.filter_poses_by_value("rfdiffusion_hotspot_contacts", args.hotspot_contacts_cutoff, ">", prefix="hotspot_cutoff", plot=True)

    # combine binder stats into composite score
    poses.calculate_composite_score(f"rfdiffusion_custom_binder_score", scoreterms=["rfdiffusion_rog", "rfdiffusion_binder_contacts"], weights=[1, -1])

    # break if diffuse_only
    if args.diffuse_only:
        logging.info(f"Diffuse only was specified, now breaking Script.")
        sys.exit(0)
    

    poses.save_poses("rfdiffusion_filtered_poses", overwrite=True)
    # setup and run partial diffusion.
    partial_diffusion_target_contig = f"B{int(args.binder_length)+1}-{int(args.binder_length) + target_length}"
    diff_opts = f"diffuser.partial_T=20 'contigmap.contigs=[{partial_diffusion_target_contig}/0 {args.binder_length}-{args.binder_length}]' 'ppi.hotspot_res=[{args.hotspot_residues.replace('A', 'B')}]'"
    logging.info("running partial diffusion")
    poses = rfdiffusion_runner.run(
        poses=poses,
        prefix="rfdiff_p",
        num_diffusions=args.num_partial_diffusions,
        options=diff_opts,
    )

    # filter down based on number of contacts
    poses.df["partial_diff_binder_contacts"] = [calc_interchain_contacts_pdb(pose, chains=["A", "B"]) for pose in poses.poses_list()]
    poses.filter_poses_by_rank(n=args.max_refinement_backbones, score_col="partial_diff_binder_contacts", ascending=False, prefix="pre_refinement_contacts", plot=True)

    # create sequences for target
    target = Poses(poses=args.target, work_dir=f"{poses.work_dir}/target_msa")
    ligandmpnn_runner.run(target, prefix="mpnn", jobstarter=gpu_jobstarter, nseq=500, model_type="soluble_mpnn")
    target_seqs = target.df["mpnn_sequence"].to_list()

    num_cycles = args.num_refinement_cycles
    fake_msa_lengths = [0, 500, 50, 25]
    for cycle in range(1, num_cycles+1):
        print(f"Starting refinement cycle {cycle}")

        # select RMSD ref_motif
        chain_selector.select("rfdiffusion_binder_res", poses=poses, chain="A")

        # prep binder input for fake-msa generation
        len_fake_msa = fake_msa_lengths[cycle]
        logging.info(f"Designing {len_fake_msa} sequences each for {len(poses)} backbones using LigandMPNN.")
        fake_msa_inputs = poses.df["poses"].to_list()
        fake_msa_poses = Poses(poses = fake_msa_inputs, work_dir = poses.work_dir + "/fake_msa_paired_seqs/")

        # store fake-msa generation poses in df for later retrieval
        poses.df[f"cycle_{cycle}_fake_msa_poses"] = fake_msa_inputs

        # design paired binder-target sequence pairs for fake MSA
        ligandmpnn_runner.run(
            poses = fake_msa_poses,
            prefix = "mpnn",
            jobstarter = optional_gpu_jobstarter,
            nseq = len_fake_msa,
            model_type = "soluble_mpnn",
            overwrite = False
        )

        # compile designed sequences into a dictionary that maps the input backbone to the corresponding fake paired msa.
        fake_msa_poses.df["prepped_seqs"] = fake_msa_poses.df["mpnn_sequence"].str.replace(":","")
        fake_paired_msa_dict = fake_msa_poses.df.groupby("input_poses")["prepped_seqs"].apply(list).to_dict()

        # copy second binder-chain into pose for tied sequence design
        chain_adder.superimpose_add_chain(
            poses = poses,
            prefix = f"cycle_{cycle}_mpnn_chain_added",
            ref_col = "poses",
            copy_chain = "A",
            jobstarter = cpu_jobstarter,
            translate_x = 250
        )

        # parse tied_residues argument for LigandMPNN
        binder_residues = poses.df["rfdiffusion_binder_res"].to_list()[0].to_list()
        symmetry_res = "|".join([f'{res},{res.replace("A", "C")}' for res in binder_residues])

        # parse symmetry_weights argument for LigandMPNN
        symmetry_weights = "|".join(["0.5,0.5" for _ in binder_residues])

        # Design binder sequences
        logging.info(f"Designing {args.num_mpnn_sequences} sequences each for {len(poses)} backbones using LigandMPNN.")
        poses = ligandmpnn_runner.run(
            poses=poses,
            prefix=f"ligandmpnn_{cycle}",
            nseq=int(args.num_mpnn_sequences),  # Use the sequences argument
            model_type="soluble_mpnn",
            options=f'--chains_to_design A,C --symmetry_residues="{symmetry_res}" --symmetry_weights="{symmetry_weights}"',
            overwrite=False
        )

        logging.info(f"removing sequence of binder before esmfold preds")
        poses = sequence_remover.run(
            poses=poses,
            prefix=f"remove_seqs_{cycle}",
            chains=[1, 2],
            overwrite=False
        )

        logging.info(f"Predicting {len(poses)} sequences with ESMFold")
        poses = esmfold_runner.run(
            poses=poses,
            prefix=f'esm_{cycle}'
        )

        # calculate RMSD, composite score and filter for ESM success
        chain_selector.select("esm_chain_A", poses=poses, chain="A")
        motif_bb_rmsd.run(poses, prefix=f"rmsd_rfdiffusion_esm_bb_{cycle}", target_motif="esm_chain_A", ref_motif="rfdiffusion_binder_res", atoms=["C", "CA", "N", "O"])

        # 1. Define the columns, titles, and y-labels before the plotting function
        cols = [f"esm_{cycle}_plddt", "rfdiff_1_plddt"]  # Replace with actual column names from poses.df
        titles = [f"esm_{cycle}_plddt", "rfdiff_1_plddt"]  # Titles for the violin plots
        y_labels = [f"esm_{cycle}_plddt", "rfdiff_1_plddt"]  # Y-axis l

        logging.info(f"plotting")
        plots.violinplot_multiple_cols(
            dataframe = poses.df,
            cols = cols, # specify the columns in poses.df that you would like to plot (as a list ["col_a", "col_b" ...]
            titles = titles, # specify the titles that the individual violins should get, as a list ["title_col_a", "title_col_b", ...] # type: ignore
            y_labels = y_labels, # y_labels for all the violins, same as above, list of labels ["...", ...]
            dims = None,
            out_path = f"{results_dir}/esmplddt_rfdiffplddt{cycle}.png", # create an output directory for your plots!
            show_fig = False # on the cluster, never show the figures! in Jupyter Notebooks you can feel free to show them.
        )

        poses.calculate_composite_score(f"cycle_{cycle}_esm_comp_score", scoreterms=[f"esm_{cycle}_plddt", f"rmsd_rfdiffusion_esm_bb_{cycle}_rmsd"], weights=[-1,1])
        poses.filter_poses_by_value(f'esm_{cycle}_plddt', 0.75, operator=">=", prefix=f"cycle_{cycle}_esm_plddt")
        poses.filter_poses_by_value(f"rmsd_rfdiffusion_esm_bb_{cycle}_rmsd", 1.5, operator="<=", prefix=f"rmsd_rfdiffusion_esm_bb_{cycle}_rmsd")

        logging.info(f"converting all pdbs to fasta files for af2 prediction")
        poses.convert_pdb_to_fasta(prefix=f"cycle_{cycle}_fasta", update_poses=True)

        # prep input for fake-msa generation
        fake_a3m_file_list = []
        os.makedirs((fake_a3m_out_dir := f"{args.output_dir}/fake_a3m_files/"), exist_ok=True)
        for pose in poses:
            msa_out_str = compile_msa_str(
                b_seq=pose[f"ligandmpnn_{cycle}_sequence"].split(":")[0],
                t_seq=target_seq,
                target_msa_sequences=target_seqs,
                paired_msa_sequences=fake_paired_msa_dict[pose[f"cycle_{cycle}_fake_msa_poses"]]
            )

            # write generated a3m string to file
            with open((outf := f"{fake_a3m_out_dir}/{pose['poses_description']}.a3m"), 'w', encoding="UTF-8") as f:
                f.write(msa_out_str)
            fake_a3m_file_list.append(outf)

        # set fake a3m files as new poses
        poses.df["poses"] = fake_a3m_file_list

        # predict with colabfold
        logging.info(f"predicting interactions with Colabfod.")
        opts = "--num-models 1 --num-recycle 3"
        poses = colabfold_runner.run(
            poses=poses,
            prefix=f"af_{cycle}",
            options=opts
        )

        # calculate interaction pAE
        calculate_poses_interaction_pae(
            f"af_{cycle}_ipAE",
            poses=poses,
            pae_list_col=f"af_{cycle}_pae_list",
            binder_start=0,
            binder_end=int(args.binder_length) - 1,
            target_start=int(args.binder_length),
            target_end=int(args.binder_length) + target_length - 1
        )

        # calculate binder-target RMSD to input from partial diffusion // previous cycle.
        dimer_rmsd_reference_poses = "rfdiff_p_location" if cycle == 1 else f"fastrelax_{cycle-1}_location"
        target_dimer_contig = f"A1-{args.binder_length},B1-{target_length}"
        ref_dimer_contig = f"A1-{args.binder_length},B{int(args.binder_length)+1}-{int(args.binder_length)+target_length}" if cycle == 1 else target_dimer_contig
        motif_bb_rmsd.run(
            poses=poses,
            prefix=f"cycle_{cycle}_dimer_bb",
            ref_col=dimer_rmsd_reference_poses,
            ref_motif=protflow.residues.from_contig(ref_dimer_contig),
            target_motif=protflow.residues.from_contig(target_dimer_contig),
            atoms=["N", "CA", "C", "O"],
            return_superimposed_poses=False
        )

        # plot ipAE and ipTM scores (for control)
        cols = [f'af_{cycle}_ipAE_pae_interaction', f'af_{cycle}_iptm']  # Replace with actual column names from poses.df
        titles = [f'af_{cycle}_ipAE_pae_interaction', f'af_{cycle}_iptm']  # Titles for the violin plots
        y_labels = [f'af_{cycle}_ipAE_pae_interaction', f'af_{cycle}_iptm']  # Y-axis l

        plots.violinplot_multiple_cols(
            dataframe = poses.df,
            cols = cols, # specify the columns in poses.df that you would like to plot (as a list ["col_a", "col_b" ...]
            titles = titles, # specify the titles that the individual violins should get, as a list ["title_col_a", "title_col_b", ...] # type: ignore
            y_labels = y_labels, # y_labels for all the violins, same as above, list of labels ["...", ...]
            dims = None,
            out_path = f"{results_dir}/filtered_ipTM_ipAE_{cycle}.png", # create an output directory for your plots!
            show_fig = False # on the cluster, never show the figures! in Jupyter Notebooks you can feel free to show them.
        )

        # calculate composite score of -> ipTM, ipAE, esm_plddt, dimer_bb_rmsd
        poses.calculate_composite_score(
            name=f"cycle_{cycle}_af2_dimer_score",
            scoreterms=[f"esm_{cycle}_plddt", f"af_{cycle}_iptm", f"af_{cycle}_ipAE_pae_interaction", f"cycle_{cycle}_dimer_bb_rmsd"],
            weights=[-1, -1, 1, 1],
            plot=True
        )

        # break on last cycle -> FastRelax ALL AF2 outputs and select based on composite score that includes interface energy!
        if cycle == num_cycles+1:
            break

        # filter af2-preds back to backbone level (after esm.)
        poses.filter_poses_by_rank(1, score_col=f"cycle_{cycle}_af2_dimer_score", remove_layers=2, prefix=f"cycle_{cycle}_af2_bb", plot=True)

        # fastrelax interface and calculate interface score.
        logging.info(f"starting_fastrelax")
        fastrelax_protocol = args.fastrelax_protocol or f"{scripts_dir}/fastrelax_sap.xml"
        rosetta_options = f"-parser:protocol {fastrelax_protocol} -beta"
        rosetta.run(
            poses=poses,
            rosetta_application="rosetta_scripts.default.linuxgccrelease",
            prefix=f"fastrelax_{cycle}",
            nstruct=15,
            options=rosetta_options
        )

        logging.info(f"Postrelax filter.")
        poses.filter_poses_by_rank(
            n=1,
            score_col=f"fastrelax_{cycle}_total_score",
            remove_layers=1,
            prefix=f"fastrelax_{cycle}_backbone",
        )

        # reindex poses back to partial diffusion index.
        poses.reindex_poses(f"cycle_{cycle}_reindex", remove_lyers=3)

    # after the last cycle fastrelax all structures.
    logging.info(f"starting_fastrelax")
    fastrelax_protocol = args.fastrelax_protocol or f"{scripts_dir}/fastrelax_sap.xml"
    rosetta_options = f"-parser:protocol {fastrelax_protocol} -beta"
    rosetta.run(
        poses=poses,
        rosetta_application="rosetta_scripts.default.linuxgccrelease",
        prefix=f"fastrelax_{cycle}",
        nstruct=15,
        options=rosetta_options
    )

    logging.info(f"Postrelax filter.")
    poses.filter_poses_by_rank(
        n=1,
        score_col=f"fastrelax_{cycle}_total_score",
        prefix=f"fastrelax_{cycle}_backbone"
    )

## This is only executed when the script is started as a script
if __name__ == "__main__":
    # Setup argument parser
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

   # Required options
    argparser.add_argument('--target', type=str, required=True, help='Path to the pdb-file that should be used as target.')
    argparser.add_argument("--output_dir", type=str, required=True, help='Path to the folder containing the output')
    argparser.add_argument("--num_mpnn_sequences", type=str, required=True, help='Number of sequences generated by LigandMPNN')
    argparser.add_argument("--jobstarter", type=str, default="slurm_gpu", help="One of {slurm_gpu, slurm_cpu, local}. Type of JobStarter that you would like to run this script.")
    argparser.add_argument("--max_gpus", type=int, default=10, help="Maximum number of GPUs to run at the same time.")
    argparser.add_argument("--max_cpus", type=int, default=320, help="Maximum number of CPUs to run at the same time.")

    # RFdiffusion Options
    argparser.add_argument("--binder_length", type=str, default="80", help='Specify the length of your binder')
    argparser.add_argument("--hotspot_residues", type=str, required=True, help='Specify the hotspot residues on your target that you want to target with your binder.')
    argparser.add_argument("--num_diff", type=int, default=100, help='Number of diffusion outputs')
    argparser.add_argument("--target_contig", type=str, help="Specify the contig of the target that should be used as a binder target")
    argparser.add_argument("--rog_cutoff", type=float, default=18, help="ROG cutoff to filter RFdiffusion outputs by. For binder lengths of 80 residues, we recommend to set this value to 14!")
    argparser.add_argument("--contacts_cutoff", type=float, default=300, help="Cutoff for interchain contact count after Rfdiffusion. For 80 residue binders, we typically use 300. The larger your binder, the larger this value can be. Check your output_dir/results/ folder for ROG and contacts distribution of your RFdiffusion output.")
    argparser.add_argument("--hotspot_contacts_cutoff", type=float, default=10, help="Minimum number of binder atoms within 4-8 Angstrom distance to hotspot residue.")
    argparser.add_argument("--rfdiff_plddt_cutoff", type=float, default=0.90, help="pLDDT cutoff to discard RFdiffusion outputs.")
    argparser.add_argument("--multiplex_rfdiffusion", type=int, help="Number of parallel copies of RFdiffusion that should run (this is an efficiency option). Be aware that the total number of diffusion outptus will be 'num_diff' * 'multiplex_rfdiffusion'.")
    argparser.add_argument("--rog_weight", type=float, default=1, help="A higher value of this weight will result in more compact structures. Too high potential weights lead to undesignable backbones. Specify the weight on the ROG portion of the custom RFdiffusion binder potential.")
    argparser.add_argument("--contacts_weight", type=float, default=1, help="A higher value of this weight will result in more contacts to the binder. Too high potential weights lead to undesignable backbones. Specify the weight on the contacts portion of the custom RFdiffusion binder potential.")
    argparser.add_argument("--num_partial_diffusions", type=int, default=3, help="Number of partial diffusion runs to run.")
    argparser.add_argument("--diffuse_only", action="store_true", help="Specify whether to exit the script after diffusion.")

    # fastrelax protocol
    argparser.add_argument("--fastrelax_protocol", default="/home/mabr3112/projects/StickyProt/scripts/fastrelax_interface_analyzer.xml", help='Specify default fastrelax_protocol to use in the Rosetta Relax steps. Put the path to your StickyProt installation\'s fastrelax script here.')

    # refinement_options
    argparser.add_argument("--num_refinement_cycles", default=3, type=int, help="Specify the number of refinement cycles of the binder design refinement you want to run after RFdiffusion. We recommend 3, usually refinement converges after latest 5 refinement cycles.")
    argparser.add_argument("--max_refinement_backbones", default=50, type=int, help="Maximum number of Backbones to refine during refinement cycles.")

    # Parse arguments
    arguments = argparser.parse_args()

    # Call the main function with parsed arguments
    main(arguments)
