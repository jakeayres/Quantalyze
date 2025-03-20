import matplotlib.pyplot as plt



def _set_ticks_inwards(fig):
    """
    Set the direction of ticks on the axes to inwards.
    
    Parameters:
    fig (matplotlib.figure.Figure): The figure to modify.
    
    Returns:
    matplotlib.figure.Figure: The modified figure.
    """
    for ax in fig.axes:
        ax.tick_params(direction='in')
    
    return fig



def nature_single_column(fig):
    """
    Adjusts the size and tick orientation of a matplotlib figure to match the 
    single-column format typically used in Nature journal publications.
    Parameters:
    fig (matplotlib.figure.Figure): The figure to be adjusted.
    Returns:
    matplotlib.figure.Figure: The adjusted figure.
    """

    fig.set_size_inches(3.5, 2.5)

    _set_ticks_inwards(fig)

    return fig


def nature_double_column(fig):
    """
    Adjusts the size of the given figure to fit a double column format for Nature journal.
    Parameters:
    fig (matplotlib.figure.Figure): The figure to be adjusted.
    Returns:
    matplotlib.figure.Figure: The adjusted figure with the new size.
    """

    fig.set_size_inches(7.2, 4.5)

    _set_ticks_inwards(fig)

    return fig