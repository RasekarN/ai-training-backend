import vectorbt as vbt

def pattern_win_rate(df, signal_col, target_factor=1.5, stop_factor=1.0):
    entries = df[signal_col] == True

    pf = vbt.Portfolio.from_signals(
        df['Close'],
        entries,
        exit_entries=entries.shift(-5, fill_value=False),
        stop_loss=stop_factor/100,
        take_profit=target_factor/100
    )

    return pf.stats()
