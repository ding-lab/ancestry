import argparse
import os

import ancestry_prediction as ap

parser = argparse.ArgumentParser()

parser.add_argument('thousand_genomes_vcf', type=str,
        help='thousand genomes vcf that is used for training.')
parser.add_argument('thousand_genomes_panel', type=str,
        help='panel file from thousand genomes that is used for training')
parser.add_argument('samples_vcf', type=str,
        help='vcf that contains samples that an ancestry prediction is to be made for.')

parser.add_argument('--output-dir', type=str,
        default=os.getcwd(), help='directory to store output files in')

args = parser.parse_args()

def main():
    ap.preform_ancestry_analysis(args.thousand_genomes_vcf, args.samples_vcf,
            args.thousand_genomes_panel, output_dir=args.output_dir)

if __name__ == '__main__':
    main()
