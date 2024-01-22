### A bidirectional Interpretable compound-protein interaction prediction framework based on cross attention

### Attention weights analysis

## obtaining attention weights
--step1-- First, disposal data format (test.txt) as follows:

+-------------------------------+------------------------------------------------------+-------+

|SMILES                         |Sequence                                              |Label  |

+-------------------------------+------------------------------------------------------+-------+

|CC1=CC=C(C=C1)N1N=C(C=C1NC(=O) |MSQERPTFYRQELNKTIWEVPERYQNLSPVGSGAYGSVCAAFDTKTGLRVAVK |1      |

|NC1=CC=C(OCCN2CCOCC2)C2=C1C=CC |KLSRPFQSIIHAKRTYRELRLLKHMKHENVIGLLDVFTPARSLEEFNDVYLVT |       |

|=C2)C(C)(C)C                   |HLMGADLNNIVKCQKLTDDHVQFLIYQILRGLKYIHSADIIHRDLKPSNLAVN |       |        

|                               |EDCELKILDFGLARHTDDEMTGYVATRWYRAPEIMLNWMHYNQTVDIWSVGCI |       |

|                               |MAELLTGRTLFPGTDHIDQLKLILRLVGTPGAELLKKISSESARNYIQSLTQMP|       |

|                               |KMNFANVFIGANPLAVDLLEKMLVLDSDKRITAAQALAHAYFAQYHDPDDEPVA|       |

|                               |DPYDQSFESRDLLIDEWKSLTYDEVISFVPPPLDQEEMES              |       |
+-------------------------------+------------------------------------------------------+-------+

--step2--  run: python preprocessing_data_test.py
--step3--  run: python get_attention_weight.py

## highlighting atoms
More details have shown in highlight_atoms.ipynb
