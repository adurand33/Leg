def st_plotly_fr(
    fig,
    *,
    height=450,
    axis_color=None,
    ticks=None,
    ticklen=10,
    grid_alpha=0.18,
    pad_ratio=0.12,
    margin_left=70,
    margin_bottom=64,        # ← add this (more room for "cm")
    x_title="cm",            # ← add: force the x title text
    y_title=None             # ← optional: set to "cm" if you want on Y too
):
    # theme colors
    theme_text = st.get_option("theme.textColor") or axis_color
    axis_color = axis_color or theme_text
    grid_color = _rgba_from_hex(axis_color, grid_alpha)

    # clone and base style
    f = go.Figure(fig)
    f.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=theme_text),
        margin=dict(l=margin_left, r=16, t=12, b=margin_bottom),  # ← more bottom margin
    )

    # x axis: keep labels, hide tick marks if you want (ticks=""), and SET TITLE
    f.update_xaxes(
        showline=False, showgrid=False, zeroline=False,
        automargin=True,
        showticklabels=True,
        ticks="", ticklen=0,                         # hide tick marks; set to "outside" + ticklen>0 if you want spacing
        tickfont=dict(color=axis_color),
        title_text=x_title,                          # ← ensure title is set
        title_font=dict(color=axis_color),           # ← title color
        title_standoff=14                            # ← space between labels and title
    )

    # y axis: horizontal grid (as you had)
    f.update_yaxes(
        showline=False,
        showgrid=True, gridcolor=grid_color, gridwidth=1,
        zeroline=False,
        automargin=True,
        showticklabels=True,
        ticks="", ticklen=0,                         # hide tick marks
        tickfont=dict(color=axis_color),
        **({"title_text": y_title, "title_font": dict(color=axis_color), "title_standoff": 14} if y_title else {})
    )