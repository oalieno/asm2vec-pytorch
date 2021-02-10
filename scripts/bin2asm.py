#!/usr/bin/env python3
import re
import os
import click
import r2pipe
import hashlib
from pathlib import Path

def sha3(data):
    return hashlib.sha3_256(data.encode()).hexdigest()

def validEXE(filename):
    magics = [bytes.fromhex('7f454c46')]
    with open(filename, 'rb') as f:
        header = f.read(4)
        return header in magics

def normalize(opcode):
    opcode = opcode.replace(' - ', ' + ')
    opcode = re.sub(r'0x[0-9a-f]+', 'CONST', opcode)
    opcode = re.sub(r'\*[0-9]', '*CONST', opcode)
    opcode = re.sub(r' [0-9]', ' CONST', opcode)
    return opcode

def fn2asm(pdf, minlen):
    # check
    if pdf is None:
        return
    if len(pdf['ops']) < minlen:
        return
    if 'invalid' in [op['type'] for op in pdf['ops']]:
        return

    ops = pdf['ops']

    # set label
    labels, scope = {}, [op['offset'] for op in ops]
    assert(None not in scope)
    for i, op in enumerate(ops):
        if op.get('jump') in scope:
            labels.setdefault(op.get('jump'), i)
    
    # dump output
    output = ''
    for op in ops:
        # add label
        if labels.get(op.get('offset')) is not None:
            output += f'LABEL{labels[op["offset"]]}:\n'
        # add instruction
        if labels.get(op.get('jump')) is not None:
            output += f' {op["type"]} LABEL{labels[op["jump"]]}\n'
        else:
            output += f' {normalize(op["opcode"])}\n'

    return output

def bin2asm(filename, opath, minlen):
    # check
    if not validEXE(filename):
        return 0
    
    r = r2pipe.open(str(filename))
    r.cmd('aaaa')

    count = 0

    for fn in r.cmdj('aflj'):
        r.cmd(f's {fn["offset"]}')
        asm = fn2asm(r.cmdj('pdfj'), minlen)
        if asm:
            uid = sha3(asm)
            asm = f''' .name {fn["name"]}
 .offset {fn["offset"]:016x}
 .file {filename.name}
''' + asm
            with open(opath / uid, 'w') as f:
                f.write(asm)
                count += 1

    print(f'[+] {filename}')

    return count

@click.command()
@click.option('-i', '--input', 'ipath', help='input directory / file', required=True)
@click.option('-o', '--output', 'opath', default='asm', help='output directory')
@click.option('-l', '--len', 'minlen', default=10, help='ignore assembly code with instructions amount smaller than minlen')
def cli(ipath, opath, minlen):
    '''
    Extract assembly functions from binary executable
    '''
    ipath = Path(ipath)
    opath = Path(opath)

    # create output directory
    if not os.path.exists(opath):
        os.mkdir(opath)

    fcount, bcount = 0, 0

    # directory
    if os.path.isdir(ipath):
        for f in os.listdir(ipath):
            if not os.path.islink(ipath / f) and not os.path.isdir(ipath / f):
                fcount += bin2asm(ipath / f, opath, minlen)
                bcount += 1
    # file
    elif os.path.exists(ipath):
        fcount += bin2asm(ipath, opath, minlen)
        bcount += 1
    else:
        print(f'[Error] No such file or directory: {ipath}')

    print(f'[+] Total scan binary: {bcount} => Total generated assembly functions: {fcount}')

if __name__ == '__main__':
    cli()
