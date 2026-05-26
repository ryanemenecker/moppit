from argparse import ArgumentParser


def run_peptiderive(pdb_path, binder_chain="B"):
    import pyrosetta

    pyrosetta.init()

    xml_string = """
    <ROSETTASCRIPTS>
        <FILTERS>
            <PeptideDeriver name="peptiderive"
            restrict_receptors_to_chains="{restrict_receptors_to_chains}"
            restrict_partners_to_chains="{restrict_partners_to_chains}"
            pep_lengths="{peptide_lengths}"
            dump_peptide_pose="true"
            dump_report_file="true"
            dump_prepared_pose="true"
            dump_cyclic_poses="true"
            skip_zero_isc="true"
            do_minimize="true"
            report_format="markdown" />
        </FILTERS>
        <PROTOCOLS>
            <Add filter_name="peptiderive"/>
        </PROTOCOLS>
    </ROSETTASCRIPTS>
    """
    peptide_length = "2,3,4,5,6,7,8,9,10,11,12,13,14,15"

    if binder_chain == "A":
        receptor = "A"
        partner = "B"
    else:
        receptor = "B"
        partner = "A"

    xml_protocol = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string(
        xml_string.format(
            restrict_receptors_to_chains=receptor,
            restrict_partners_to_chains=partner,
            peptide_lengths=peptide_length,
        )
    )

    peptiderive_from_xml_protocol = xml_protocol.get_mover("ParsedProtocol")
    pose = pyrosetta.pose_from_file(pdb_path)
    peptiderive_from_xml_protocol.apply(pose)


def main(args):
    run_peptiderive(args.pdb, args.binder_chain)


def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--pdb", required=True, type=str, help="The path to the binder-target protein complex structure")
    parser.add_argument("--binder_chain", default="B", choices=["A", "B"], type=str,
                        help="Whether the binder is chain A or chain B in the protein complex structure")
    return parser


def main_cli(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    main(args)


if __name__ == "__main__":
    main_cli()