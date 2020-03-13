import argparse
import logging
import os

import pandas as pd

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

CHROM_INDEX = 0
POS_INDEX = 1
REF_INDEX = 2
DEPTH_INDEX = 3
BASE_INDEX = 4

parser = argparse.ArgumentParser()

parser.add_argument('--readcount-dir', type=str,
        help='directory with readcount files to be turned into vcf. Sample names will be obtained \
from readcount filename')
parser.add_argument('--genomes-vcf', type=str,
        help='thousand genomes vcf with sites for ancestry anlysis')
parser.add_argument('--output', type=str,
        help='output filepath')

args = parser.parse_args()

def get_readcount_fps_from_dir(dir_path, readcount_extension=True):
    fps = os.listdir(dir_path)
    
    if readcount_extension:
        fps = [fp for fp in fps
              if '.readcount' in fp]
    
    return [os.path.join(dir_path, fp) for fp in fps]

def get_headerless_vcf_df(f_obj, stop_on='#CHROM'):
    line = ''
    header = ''
    while True:
        line = f_obj.readline()
        header += line
        if line[:6] == stop_on:
            break
            
    df = pd.read_csv(f_obj, sep='\t', header=None)
    df.columns = line[1:].replace('""', '').replace('\n', '').split('\t')

    return df, header

def get_headerless_vcf_position_tups(f_obj, stop_on='#CHROM', add_chr=True):
    header = ''
    in_header = True
    tups = []
    for line in f_obj:
        if in_header:
            header += line
            if line[:6] == stop_on:
                in_header=False
        else:
            pieces = line.split('\t', 5)
            chrom = pieces[0]
#            if add_chr and 'chr' not in chrom:
#                chrom = 'chr' + chrom
            
            pos = pieces[1]
            ref = pieces[3]
            alt = pieces[4]
            
            
            tups.append((chrom, pos, ref, alt))

    return tups, header

def breakdown_readcount_line(line, add_chr=True):
    pieces = line.strip().split('\t')
    
    chrom = pieces[CHROM_INDEX]
#    if add_chr and 'chr' not in chrom:
#        chrom = 'chr' + chrom
        
    pos = pieces[POS_INDEX]
    ref = pieces[REF_INDEX]
    depth = int(pieces[DEPTH_INDEX])
    
    d = {}
    for base_chunk in pieces[BASE_INDEX:]:
        ps = base_chunk.split(':', 2)
        d[ps[0]] = int(ps[1])
    
    return chrom, pos, ref, depth, d
    
    
def get_readcount_dict(readcount_fp):
    """
    readcount_dict - {(chr:pos:ref): {base_dict: {a:40....}, depth: 60}}
    """
    f = open(readcount_fp)
    readcount_dict = {}
    for line in f:
        chrom, pos, ref, depth, base_dict = breakdown_readcount_line(line)
        readcount_dict[(chrom, pos, ref)] = {
            'base_dict': base_dict,
            'depth': depth
        }
    f.close()
        
    return readcount_dict

def call_position(depth, ref_count, alt_count, min_depth=8, m=1):
    if depth < min_depth * m:
            return '.|.'

    if ref_count is None or alt_count is None:
        return '.|.'

    # check hom ref
    if ref_count >= 8 * m and alt_count < 4 * m:
        return '0|0'
    # check het
    if ref_count >= 4 * m and alt_count >= 4 * m:
        return '0|1'
    # check hom alt
    if alt_count >= 8 * m and ref_count < 4 * m:
        return '1|1'

    return '.|.'

def call_positions(genomes_position_tups, readcount_dict):
    """
    returns (chrom, pos, ref, alt, call)
    """

    calls = []
    for chrom, pos, ref, alt in genomes_position_tups:
        d = readcount_dict.get((chrom, pos, ref))
        if d is not None:
            base_dict = d['base_dict']
            depth = d['depth']
            call = call_position(depth, base_dict.get(ref), base_dict.get(alt))
            calls.append((chrom, pos, ref, alt, call))
        else:
            calls.append((chrom, pos, ref, alt, '.|.'))
            
    return calls

def merge_sample_calls(sample_to_calls):
    sample_columns = {s:[] for s in sample_to_calls.keys()}
    
    vcf_columns = {
        'CHROM': [],
        'POS': [],
        'ID': [],
        'REF': [],
        'ALT': [],
        'QUAL': [],
        'FILTER': [],
        'INFO': [],
        'FORMAT': [],
    }
    
    for chrom, pos, ref, alt, _ in list(sample_to_calls.values())[0]:
        vcf_columns['CHROM'].append(chrom)
        vcf_columns['POS'].append(pos)
        vcf_columns['ID'].append('.')
        vcf_columns['REF'].append(ref)
        vcf_columns['ALT'].append(alt)
        vcf_columns['QUAL'].append('.')
        vcf_columns['FILTER'].append('PASS')
        vcf_columns['INFO'].append('.')
        vcf_columns['FORMAT'].append('GT')

    for sample, calls in sample_to_calls.items():
        sample_columns[sample] = [call for _, _, _, _, call in calls]
        
    sample_columns.update(vcf_columns)
    
    df = pd.DataFrame.from_dict(sample_columns)
    sample_cols = sorted([s for s in sample_to_calls.keys()])
    df = df[['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT'] + sample_cols]
    
    return df

def write_vcf(header, vcf_df, fp):
    header_lines = header.strip().split('\n')
    if '#CHROM' in header_lines[-1]:
        header = '\n'.join(header_lines[:-1])
    
    df_str = vcf_df.to_csv(None, sep='\t', index=False, header=False)
    fields_str = '#' + '\t'.join(vcf_df.columns) + '\n'
    to_write = header + '\n' + fields_str + df_str
    
    f = open(fp, 'w')
    f.write(to_write)
    f.close()


if __name__ == '__main__':
    fps = get_readcount_fps_from_dir(args.readcount_dir)

    logging.info('grabbing positions and header from thousand genomes vcf')
    genomes_position_tups, genomes_header = get_headerless_vcf_position_tups(open(args.genomes_vcf))
    logging.info(f'getting readcount dictionaries for {len(fps)} input readcount files')
    readcount_dicts = [get_readcount_dict(fp) for fp in fps]

    logging.info('making calls')
    sample_to_calls = {s.split('/')[-1].replace('.readcount', ''):call_positions(genomes_position_tups, readcount_dict) 
                   for s, readcount_dict in zip(fps, readcount_dicts)}

    logging.info('merging sample call dataframes')
    new_df = merge_sample_calls(sample_to_calls)
    logging.info(f'call dataframe of size {new_df.shape} created')

    logging.info('writing vcf')
    write_vcf(genomes_header, new_df, args.output)
