"""
Draw Entity-Relationship Diagram (ERD) for Chinook Database and save to figures/erd.png.
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import get_connection, get_schema_summary


def draw_erd():
    conn = get_connection()
    schema = get_schema_summary(conn)
    conn.close()

    fig, ax = plt.subplots(figsize=(20, 13), dpi=300)
    ax.set_facecolor("#0F172A")  # Modern dark background
    fig.patch.set_facecolor("#0F172A")

    # Layout coordinates for each table (x, y, width, height)
    # Centered around the core purchase flow: Customer -> Invoice -> InvoiceLine -> Track -> Album -> Artist
    coords = {
        "Customer": (1.0, 7.5, 2.5, 3.4, "#3B82F6"),       # Blue
        "Employee": (1.0, 2.5, 2.5, 3.6, "#6366F1"),       # Indigo
        "Invoice": (4.5, 7.5, 2.5, 3.2, "#0EA5E9"),        # Sky
        "InvoiceLine": (8.0, 7.5, 2.5, 2.8, "#10B981"),    # Emerald (Core Fact)
        "Track": (11.5, 7.0, 2.6, 4.4, "#F59E0B"),         # Amber (Core Dimension)
        "Album": (15.0, 7.5, 2.4, 2.5, "#EC4899"),         # Pink
        "Artist": (18.2, 7.5, 2.2, 2.2, "#8B5CF6"),        # Purple
        "Genre": (11.5, 2.5, 2.2, 2.2, "#14B8A6"),         # Teal
        "MediaType": (14.5, 3.0, 2.4, 2.2, "#EAB308"),     # Yellow
        "PlaylistTrack": (8.0, 2.5, 2.5, 2.2, "#64748B"), # Slate (Excluded from clustering)
        "Playlist": (4.8, 2.5, 2.2, 2.2, "#64748B")       # Slate (Excluded from clustering)
    }

    # Render each entity box
    for table_name, (x, y, w, h, color) in coords.items():
        info = schema[table_name]
        cols = info["columns"]
        row_cnt = info["row_count"]

        # Outer rounded box
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.08,rounding_size=0.15",
            edgecolor=color, facecolor="#1E293B", linewidth=2.0, zorder=2
        )
        ax.add_patch(box)

        # Header background
        header_h = 0.55
        header_box = patches.FancyBboxPatch(
            (x, y + h - header_h), w, header_h,
            boxstyle="round,pad=0.08,rounding_size=0.15",
            edgecolor=color, facecolor=color, linewidth=1.5, zorder=3
        )
        ax.add_patch(header_box)

        # Header text
        ax.text(
            x + w / 2, y + h - header_h / 2,
            f"{table_name.upper()} ({row_cnt:,})",
            color="#FFFFFF", fontsize=9.5, fontweight="bold",
            ha="center", va="center", zorder=4
        )

        # Columns list (max 10 to fit cleanly)
        line_y = y + h - header_h - 0.25
        displayed_cols = cols[:8]
        for c in displayed_cols:
            cname = c["name"]
            prefix = "[PK] " if c["pk"] else ("[FK] " if any(fk["from_col"] == cname for fk in info["foreign_keys"]) else "     ")
            col_text = f"{prefix}{cname}: {c['type'][:8]}"
            text_color = "#F8FAFC" if c["pk"] else ("#93C5FD" if "[FK]" in prefix else "#94A3B8")
            ax.text(x + 0.15, line_y, col_text, color=text_color, fontsize=7.2, va="center", zorder=4)
            line_y -= 0.30

        if len(cols) > 8:
            ax.text(x + 0.15, line_y, f"  ... ({len(cols) - 8} more cols)", color="#64748B", fontsize=6.8, va="center", zorder=4)

    # Relationship connectors (From_Table, To_Table, Start_pt, End_pt, label, color)
    relationships = [
        # Employee -> Employee (self-referencing ReportsTo)
        ("Employee", "Employee", (1.0, 5.0), (1.0, 5.5), "ReportsTo (1:N)", "#6366F1"),
        # Customer -> Employee (SupportRepId)
        ("Customer", "Employee", (2.2, 7.5), (2.2, 6.1), "1:N (SupportRep)", "#6366F1"),
        # Customer -> Invoice
        ("Customer", "Invoice", (3.5, 9.0), (4.5, 9.0), "1:N", "#3B82F6"),
        # Invoice -> InvoiceLine
        ("Invoice", "InvoiceLine", (7.0, 9.0), (8.0, 9.0), "1:N", "#0EA5E9"),
        # Track -> InvoiceLine
        ("Track", "InvoiceLine", (11.5, 9.0), (10.5, 9.0), "1:N (Purchased)", "#10B981"),
        # Album -> Track
        ("Album", "Track", (15.0, 8.8), (14.1, 8.8), "1:N", "#EC4899"),
        # Artist -> Album
        ("Artist", "Album", (18.2, 8.8), (17.4, 8.8), "1:N", "#8B5CF6"),
        # Genre -> Track
        ("Genre", "Track", (12.6, 4.7), (12.6, 7.0), "1:N", "#14B8A6"),
        # MediaType -> Track
        ("MediaType", "Track", (14.5, 4.8), (13.5, 7.0), "1:N", "#EAB308"),
        # Playlist -> PlaylistTrack
        ("Playlist", "PlaylistTrack", (7.0, 3.6), (8.0, 3.6), "1:N", "#64748B"),
        # Track -> PlaylistTrack
        ("Track", "PlaylistTrack", (11.5, 7.2), (10.5, 4.5), "1:N", "#64748B")
    ]

    for from_tbl, to_tbl, start, end, label, rel_color in relationships:
        if from_tbl == to_tbl:
            # Self-reference curve
            arc = patches.Arc((0.8, 5.25), 0.6, 0.7, angle=0, theta1=90, theta2=270, color=rel_color, lw=1.5, ls="--")
            ax.add_patch(arc)
            ax.text(0.35, 5.25, label, color=rel_color, fontsize=7.0, va="center", ha="right")
        else:
            ax.annotate(
                "", xy=end, xytext=start,
                arrowprops=dict(
                    arrowstyle="->,head_width=0.35,head_length=0.45",
                    color=rel_color, lw=1.6, ls="-" if "#64748B" not in rel_color else "--"
                ),
                zorder=1
            )
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2 + 0.15
            ax.text(
                mid_x, mid_y, label,
                color=rel_color, fontsize=7.2, fontweight="bold",
                ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#0F172A", edgecolor="none", alpha=0.85),
                zorder=5
            )

    # Title and Legend / Note
    ax.text(
        10.5, 12.3, "CHINOOK DATABASE - ENTITY RELATIONSHIP DIAGRAM (ERD)",
        color="#F8FAFC", fontsize=16, fontweight="bold", ha="center"
    )
    ax.text(
        10.5, 11.85, "Trục quan hệ chính: Customer ──> Invoice ──> InvoiceLine ──> Track ──> Album ──> Artist",
        color="#38BDF8", fontsize=10.5, ha="center"
    )
    ax.text(
        10.5, 0.8,
        "Ghi chú: Bảng Playlist & PlaylistTrack (viền nét đứt màu xám) là danh mục do cửa hàng tạo sẵn, không liên kết trực tiếp với CustomerId.\n"
        "Do đó, cụm khách hàng sẽ dựa trên hành vi mua thực tế tại InvoiceLine (Track, Album, Genre, Artist).",
        color="#94A3B8", fontsize=9.0, ha="center", style="italic",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#1E293B", edgecolor="#334155", lw=1.2)
    )

    ax.set_xlim(-0.2, 21.0)
    ax.set_ylim(0.0, 13.0)
    ax.axis("off")

    os.makedirs("figures", exist_ok=True)
    plt.tight_layout()
    output_path = "figures/erd.png"
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Successfully generated {output_path}!")


if __name__ == "__main__":
    draw_erd()
