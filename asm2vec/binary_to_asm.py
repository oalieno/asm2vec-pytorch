import re
import os
import hashlib
import r2pipe
import logging
from pathlib import Path


def _sha3(asm: str) -> str:
    """Produces SHA3 for each assembly function
    :param asm: input assembly function
    """
    return hashlib.sha3_256(asm.encode()).hexdigest()


def _valid_exe(filename: str) -> bool:
    """Extracts magic bytes and returns the header
    :param filename: name of the malware file (SHA1)
    :return: Boolean of the header existing in magic bytes
    """
    magics = [bytes.fromhex('cffaedfe')]
    with open(filename, 'rb') as f:
        header = f.read(4)
        return header in magics


def _normalize(opcode: str) -> str:
    """ Normalizes the input string
    :param opcode: opcode of the binary
    """
    opcode = opcode.replace(' - ', ' + ')
    opcode = re.sub(r'0x[0-9a-f]+', 'CONST', opcode)
    opcode = re.sub(r'\*[0-9]', '*CONST', opcode)
    opcode = re.sub(r' [0-9]', ' CONST', opcode)
    return opcode


def _fn_to_asm(pdf: dict | None, asm_minlen: int) -> str:
    """Converts functions to assembly code
    :param pdf: disassembly
    :param asm_minlen: minimum length of assembly functions to be extracted
    """
    if pdf is None:
        return ''
    if len(pdf['ops']) < asm_minlen:
        return ''
    if 'invalid' in [op['type'] for op in pdf['ops']]:
        return ''

    ops = pdf['ops']

    labels, scope = {}, [op['offset'] for op in ops]
    assert (None not in scope)
    for i, op in enumerate(ops):
        if op.get('jump') in scope:
            labels.setdefault(op.get('jump'), i)

    output = ''
    for op in ops:
        if labels.get(op.get('offset')) is not None:
            output += f'LABEL{labels[op["offset"]]}:\n'
        if labels.get(op.get('jump')) is not None:
            output += f' {op["type"]} LABEL{labels[op["jump"]]}\n'
        else:
            output += f' {_normalize(op["opcode"])}\n'

    return output


def bin_to_asm(filename: Path, output_path: Path, asm_minlen: int) -> int:
    """Fragments the input binary into assembly functions via r2pipe
    :param filename: name of the malware file  (SHA1)
    :param output_path: path to the folder to store the assembly functions for each malware
    :param asm_minlen: the minimum length of assembly functions to be extracted
    :return: the number of assembly functions
    """
    if not _valid_exe(filename):
        logging.info('The input file is invalid.')
        return 0

    r = r2pipe.open(str(filename))
    r.cmd('aaaa')

    count = 0

    for fn in r.cmdj('aflj'):
        r.cmd(f's {fn["offset"]}')
        asm = _fn_to_asm(r.cmdj('pdfj'), asm_minlen)
        if asm:
            uid = _sha3(asm)
            asm = f''' .name {fn["name"]}\
            .offset {fn["offset"]:016x}\
            .file {filename.name}''' + asm
            output_asm = os.path.join(output_path, uid)
            with open(output_asm, 'w') as file:
                file.write(asm)
                count += 1
    return count


def convert_to_asm(input_path, output_path, minlen_upper: int, minlen_lower: int) -> list:
    """ Extracts assembly functions from malware files and saves them
    into separate folder per binary
    :param input_path: the path to the malware binaries
    :param output_path: the path for the assembly functions to be extracted
    :param minlen_upper: The minimum number of assembly functions needed for disassembling
    :param minlen_lower: If disassembling not possible with with minlen_upper, lower the minimum number
    of assembly functions to minlen_lower
    :return: List of sha1 of disassembled malware files
    """

    binary_dir = Path(input_path)
    asm_dir = Path(output_path)

    if not os.path.exists(asm_dir):
        os.mkdir(asm_dir)

    function_count, binary_count, not_found = 0, 0, 0
    disassembled_bins = []

    if os.path.isdir(binary_dir):
        for entry in os.scandir(binary_dir):
            out_dir = os.path.join(asm_dir, entry.name)
            if not (os.path.exists(out_dir)):
                os.mkdir(out_dir)
            function_count += bin_to_asm(Path(entry), Path(out_dir), minlen_upper)
            if function_count == 0:
                function_count += bin_to_asm(Path(entry), Path(out_dir), minlen_lower)
                if function_count == 0:
                    os.rmdir(out_dir)
                    logging.info('The binary {} was not disassembled'.format(entry.name))
                else:
                    binary_count += 1
                    disassembled_bins.append(entry.name)
            else:
                binary_count += 1
                disassembled_bins.append(entry.name)
    else:
        not_found += 1
        logging.info("[Error] No such file or directory: {}".format(binary_dir))

    logging.info("Total scanned binaries: {}".format(binary_count))
    logging.info("Not converted binaries: {}".format(not_found))

    return disassembled_bins
