import numpy as np

from .swings import SwingDetector
from .structure import StructureEngine
from .classical_patterns import ClassicalPatterns

class PatternEngine:
    def __init__(self):
        self.swings = SwingDetector()
        self.structure = StructureEngine()
        self.patterns = ClassicalPatterns()

    def process(self, df):
        df = self.swings.detect(df)
        df = self.structure.detect_bos(df)
        df = self.patterns.detect_double_top(df)

        return df
