import pandas as pd

class SwingDetector:
    def __init__(self, left=3, right=3):
        self.left = left
        self.right = right

    def detect(self, df):
        df = df.copy()
        df['swing_high'] = False
        df['swing_low'] = False

        for i in range(self.left, len(df)-self.right):
            high = df['High'].iloc[i]
            low = df['Low'].iloc[i]

            if high == max(df['High'].iloc[i-self.left:i+self.right+1]):
                df.at[df.index[i], 'swing_high'] = True
            
            if low == min(df['Low'].iloc[i-self.left:i+self.right+1]):
                df.at[df.index[i], 'swing_low'] = True
        
        return df
