"""Shared Fig. 1 layout and local vector-only rebuild.

Run: python paper/figures/fig1_layout.py
The cached fig1_data_panels.{pdf,svg} preserve the original A/C vectors from
Fig. 1 before the 2026-10-04 layout revision. No curves are inferred or refitted.
For regeneration from MD/model arrays, make_fig1.py uses the same drawing code.
"""
from pathlib import Path
from copy import deepcopy
import re
import subprocess
import tempfile
from xml.etree import ElementTree as ET

W_MM, H_MM = 183, 96


def draw_layout(ax):
    """Draw panel headings and the central functional / training workflow in mm."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    ink, gray, blue, rust = '#252525', '#777770', '#386fae', '#b95739'
    x, w, mid = 65.5, 54, 92.5

    def text(px, py, label, size=6.2, color=ink, weight='normal', ha='center'):
        return ax.text(px, py, label, fontsize=size, color=color, weight=weight,
                       ha=ha, va='center', zorder=4)

    def card(y, h, fill, edge):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle='round,pad=0,rounding_size=1', facecolor=fill,
                     edgecolor=edge, linewidth=.65, zorder=2))

    def arrow(x0, y0, x1, y1, color=gray):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>',
                     mutation_scale=7, linewidth=.75, color=color,
                     shrinkA=1, shrinkB=1, zorder=3))

    for px, letter, title in [(4, 'A', 'Simulation data'),
                              (65.5, 'B', 'Functional and training'),
                              (127, 'C', 'Predictions')]:
        text(px, 92, letter, 8, weight='bold', ha='left')
        text(px+4.1, 92, title, 7, weight='bold', ha='left')

    text(mid, 84.7, r'Ion densities $n_+(z),\ n_-(z)$', 6.6)
    arrow(mid, 81.5, mid, 77.7)

    card(65, 12, '#f3f6fa', '#91abc9')
    text(mid, 73.6, 'Learned radial convolutions', 6.2, blue)
    text(mid, 68.3, r'$h_{m\alpha}=K_m\star n_\alpha$', 7.1)
    arrow(mid, 64.4, mid, 60.7)

    card(45, 15, '#eaf0f8', '#7398c4')
    text(mid, 56.3, 'Scalar excess free energy', 6.3, blue, 'bold')
    text(mid, 49.6,
         r'$F_{\rm ex}^{\theta}=F_{\rm pair}+\int\Phi_\theta(\mathbf{h})\,\mathrm{d}\mathbf{r}$', 6.4)
    arrow(mid, 44.5, mid, 41.3)

    text(mid, 38.5, 'Add analytic ideal and Coulomb terms', 5.8, gray)
    text(mid, 32.9, r'$F=F_{\rm id}+F_{\rm Coul}+F_{\rm ex}^{\theta}$', 7)
    arrow(mid, 29.7, mid, 25.4)

    card(7.5, 17.2, '#faf0eb', '#c7826c')
    text(mid, 21.5, 'Force matching', 6.5, rust, 'bold')
    text(mid, 15.9,
         r'$\mu_\alpha^{\rm int}=\delta(F_{\rm Coul}+F_{\rm ex}^{\theta})/\delta n_\alpha$', 5.9)
    text(mid, 10.7,
         r'$f_\alpha^{\theta}=-n_\alpha\partial_z\mu_\alpha^{\rm int}\quad\longleftrightarrow\quad f_\alpha^{\rm MD}$', 6.2)
    # The measured force profile feeds the loss; the scalar F feeds prediction.
    arrow(59.2, 16, x-.7, 16, rust)
    arrow(x+w+.8, 33, 124.7, 33, blue)

    text(128, 88.3, r'Density profiles from $V(z)$', 6, ha='left')
    text(128, 51, 'Structure and number response', 6, ha='left')


def rebuild_from_vectors():
    """Combine unchanged A/C vectors with new B, exporting PDF/SVG/PNG."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pypdf import PdfReader, PdfWriter, PageObject
    from pypdf.generic import DecodedStreamObject, NameObject

    root=Path(__file__).resolve().parent/'fig1'
    plt.rcParams.update({'font.family':'sans-serif',
        'font.sans-serif':['Arial','DejaVu Sans'], 'mathtext.fontset':'dejavusans',
        'svg.fonttype':'none','pdf.fonttype':42})
    fig=plt.figure(figsize=(W_MM/25.4,H_MM/25.4))
    ax=fig.add_axes([0,0,1,1]); ax.set(xlim=(0,W_MM),ylim=(0,H_MM)); ax.axis('off')
    draw_layout(ax)
    with tempfile.TemporaryDirectory(prefix='ndft-fig1-') as td:
        tmp=Path(td)
        fig.savefig(tmp/'layout.pdf',transparent=True)
        fig.savefig(tmp/'layout.svg',transparent=True)
        plt.close(fig)
        data_page=PdfReader(root/'fig1_data_panels.pdf').pages[0]
        width,height=float(data_page.mediabox.width),float(data_page.mediabox.height)
        page=PageObject.create_blank_page(width=width,height=height)
        background=DecodedStreamObject()
        background.set_data(f'q 1 1 1 rg 0 0 {width} {height} re f Q\n'.encode())
        page[NameObject('/Contents')]=background
        page.merge_page(data_page)
        page.merge_page(PdfReader(tmp/'layout.pdf').pages[0])
        writer=PdfWriter(); writer.add_page(page)
        writer.add_metadata({'/Title':'Figure 1: Simulation data, functional and training, predictions'})
        with (root/'fig1_schematic.pdf').open('wb') as f: writer.write(f)

        ns='http://www.w3.org/2000/svg'
        ET.register_namespace('',ns)
        ET.register_namespace('xlink','http://www.w3.org/1999/xlink')
        base=ET.parse(root/'fig1_data_panels.svg').getroot()
        overlay=ET.parse(tmp/'layout.svg').getroot()
        vb=base.get('viewBox').split()
        base.insert(0,ET.Element('{'+ns+'}rect',{'x':'0','y':'0','width':vb[2],'height':vb[3],'fill':'white'}))
        # Prefix generated IDs to avoid collisions with the cached plot markers/clip paths.
        mapping={el.get('id'):'layout_'+el.get('id') for el in overlay.iter() if el.get('id')}
        for el in overlay.iter():
            if el.get('id'): el.set('id',mapping[el.get('id')])
            for key,value in list(el.attrib.items()):
                value=re.sub(r'url\(#([^)]*)\)',lambda m:'url(#'+mapping.get(m[1],m[1])+')',value)
                if value.startswith('#') and value[1:] in mapping:value='#'+mapping[value[1:]]
                el.set(key,value)
        for child in overlay:
            if child.tag.split('}')[-1] in ('g','defs'):base.append(deepcopy(child))
        ET.ElementTree(base).write(root/'fig1_schematic.svg',encoding='utf-8',xml_declaration=True)
        svg_path=root/'fig1_schematic.svg'
        svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines())+'\n')
    # Render the actual delivered vector PDF, ensuring preview and manuscript agree.
    subprocess.run(['pdftoppm','-singlefile','-r','300','-png',
                    str(root/'fig1_schematic.pdf'),str(root/'fig1_schematic')],check=True)
    print('Updated fig1_schematic.pdf, .svg and .png from preserved data vectors.')


if __name__=='__main__':
    rebuild_from_vectors()
