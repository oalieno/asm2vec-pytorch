from pathlib import Path
from unittest import TestCase
from asm2vec.binary_to_asm import (bin_to_asm,
                                   convert_to_asm,
                                   _fn_to_asm,
                                   _normalize,
                                   _sha3,
                                   _valid_exe)


class TestBinaryToAsm(TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n--- TestBinaryToAsm ---")
        cls.output_path = 'malware_asm/'
        cls.pdf_dict = {'name': 'main', 'size': 18, 'addr': 4294974144,
                        'ops': [{'offset': 4294974144, 'esil': 'rbp,8,rsp,-,=[8],8,rsp,-=', 'refptr': 0,
                                 'fcn_addr': 4294974144, 'fcn_last': 4294974161, 'size': 1, 'opcode': 'push rbp',
                                 'disasm': 'push rbp', 'bytes': '55', 'family': 'cpu', 'type': 'rpush',
                                 'reloc': 'False', 'type_num': 268435468, 'type2_num': 0,
                                 'flags': ['main', 'entry0', 'section.0.__TEXT.__text', 'sym.func.100001ac0', 'rip'],
                                 'comment': 'WzAwXSAtci14IHNlY3Rpb24gc2l6ZSA3Mzc2IG5hbWVkIDAuX19URVhULl9fdGV4dA=='},
                                {'offset': 4294974145, 'esil': 'rsp,rbp,=', 'refptr': 0, 'fcn_addr': 4294974144,
                                 'fcn_last': 4294974159, 'size': 3, 'opcode': 'mov rbp, rsp', 'disasm': 'mov rbp, rsp',
                                 'bytes': '4889e5', 'family': 'cpu', 'type': 'mov', 'reloc': 'False', 'type_num': 9,
                                 'type2_num': 0}, {'offset': 4294974148, 'esil': 'rbx,8,rsp,-,=[8],8,rsp,-=',
                                                   'refptr': 0, 'fcn_addr': 4294974144, 'fcn_last': 4294974161,
                                                   'size': 1, 'opcode': 'push rbx', 'disasm': 'push rbx', 'bytes': '53',
                                                   'family': 'cpu', 'type': 'rpush', 'reloc': 'False',
                                                   'type_num': 268435468, 'type2_num': 0},
                                {'offset': 4294974149, 'esil': 'rax,8,rsp,-,=[8],8,rsp,-=', 'refptr': 0,
                                 'fcn_addr': 4294974144, 'fcn_last': 4294974161, 'size': 1, 'opcode': 'push rax',
                                 'disasm': 'push rax', 'bytes': '50', 'family': 'cpu', 'type': 'rpush',
                                 'reloc': 'False', 'type_num': 268435468, 'type2_num': 0},
                                {'offset': 4294974150, 'esil': 'rsi,rbx,=', 'refptr': 0, 'fcn_addr': 4294974144,
                                 'fcn_last': 4294974159, 'size': 3, 'opcode': 'mov rbx, rsi', 'disasm': 'mov rbx, rsi',
                                 'bytes': '4889f3', 'family': 'cpu', 'type': 'mov', 'reloc': 'False', 'type_num': 9,
                                 'type2_num': 0}, {'offset': 4294974153, 'ptr': 4294985864,
                                                   'esil': '0x2db8,rip,+,[8],rax,=', 'refptr': 8,
                                                   'fcn_addr': 4294974144, 'fcn_last': 4294974155, 'size': 7,
                                                   'opcode': 'mov rax, qword [rip + 0x2db8]',
                                                   'disasm': 'mov rax, qword [0x100004888]', 'bytes': '488b05b82d0000',
                                                   'family': 'cpu', 'type': 'mov', 'reloc': 'False', 'type_num': 9,
                                                   'type2_num': 0, 'refs': [{'addr': 4294985864, 'type': 'DATA',
                                                                             'perm': 'r--'}]}, {'offset': 4294974160,
                                                                                                'esil': 'rax,rip,=',
                                                                                                'refptr': 0,
                                                                                                'fcn_addr': 4294974144,
                                                                                                'fcn_last': 4294974160,
                                                                                                'size': 2,
                                                                                                'opcode': 'jmp rax',
                                                                                                'disasm': 'jmp rax',
                                                                                                'bytes': 'ffe0',
                                                                                                'family': 'cpu',
                                                                                                'type': 'rjmp',
                                                                                                'reloc': 'False',
                                                                                                'type_num': 268435458,
                                                                                                'type2_num': 0}]}

    def test_sha3(self):
        """Should return 64-character long string"""
        asm = ("push rbp\n"
               "mov rbp, rsp\n"
               "push rbx\n"
               "push rax\n"
               "mov rbx, rsi\n"
               "mov rax, qword [rip + CONST]\n"
               "jmp rax")
        self.assertRegex(_sha3(asm), '^[a-f0-9]{64}')

    def test_valid_exe_when_valid_magic_bytes(self):
        """Should return boolean"""
        binary_location = "malware_bin/5cca32eb8f9c2a024a57ce12e3fb66070662de80"
        filename = Path(binary_location)
        magic_bytes = ['cffaedfe']
        self.assertEqual(_valid_exe(filename, magic_bytes), True)

    def test_valid_exe_when_not_valid_magic_bytes(self):
        """Should return boolean"""
        binary_location = "malware_bin/5cca32eb8f9c2a024a57ce12e3fb66070662de80"
        filename = Path(binary_location)
        magic_bytes = ['cafebabe']
        self.assertEqual(_valid_exe(filename, magic_bytes), False)

    def test_normalize_when_offset(self):
        """Should return normalized opcode"""
        opcode = "mov rax, qword [rip + 0x2db8]"
        expected_norm_opcode = "mov rax, qword [rip + CONST]"
        self.assertEqual(_normalize(opcode), expected_norm_opcode)

    def test_normalize_when_no_offset(self):
        """Should return normalized opcode"""
        opcode = 'mov rbx, rsi'
        expected_norm_opcode = "mov rbx, rsi"
        self.assertEqual(_normalize(opcode), expected_norm_opcode)

    def test_fn_to_asm_returns_empty_string_when_pdf_none(self):
        """Should return assembly functions with normalized opcode"""
        pdf = None
        asm_min = 5
        expected_asm = ""
        self.assertEqual(_fn_to_asm(pdf, asm_min), expected_asm)

    def test_fn_to_asm_returns_empty_string_when_pdfops_shorter_than_minlen(self):
        """Should return assembly functions with normalized opcode"""
        asm_minlen = 10
        expected_asm = ""
        self.assertEqual(_fn_to_asm(self.pdf_dict, asm_minlen), expected_asm)

    def test_fn_to_asm_returns_expected_asm(self):
        """Should return assembly functions with normalized opcode"""
        asm_min = 5
        expected_asm = (" push rbp\n"
                        " mov rbp, rsp\n"
                        " push rbx\n"
                        " push rax\n"
                        " mov rbx, rsi\n"
                        " mov rax, qword [rip + CONST]\n"
                        " jmp rax\n")
        self.assertEqual(_fn_to_asm(self.pdf_dict, asm_min), expected_asm)

    def test_bin_to_asm_returns_expected_number_of_disassembled_files(self):
        binary_location = "malware_bin/5cca32eb8f9c2a024a57ce12e3fb66070662de80"
        asm_minlen = 5
        magic_bytes = ['cffaedfe']
        self.assertEqual(bin_to_asm(Path(binary_location), Path(self.output_path), asm_minlen, magic_bytes), 1)

    def test_bin_to_asm_returns_expected_number_of_disassembled_files_when_pdfops_shorter_than_minlen(self):
        binary_location = "malware_bin/5cca32eb8f9c2a024a57ce12e3fb66070662de80"
        asm_minlen = 10
        magic_bytes = ['cffaedfe']
        self.assertEqual(bin_to_asm(Path(binary_location), self.output_path, asm_minlen, magic_bytes), 0)

    def test_convert_to_asm_returns_expected_sha1(self):
        input_path = 'malware_bin/'
        asm_minlen_upper = 10
        asm_minlen_lower = 5
        expected_sha1 = ["5cca32eb8f9c2a024a57ce12e3fb66070662de80"]
        self.assertEqual(convert_to_asm(input_path, self.output_path, asm_minlen_upper, asm_minlen_lower),
                         expected_sha1)
