% Single-column preprint, using George Kour's MIT-licensed arxiv-style.
% Upstream: https://github.com/kourgeorge/arxiv-style
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{preprint}[2026/10/07 Single-column scientific preprint]
\LoadClass[10pt,letterpaper,oneside]{article}
\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage[english]{babel}
\RequirePackage{arxiv}
\RequirePackage{amsmath,amssymb}
\RequirePackage{newtxtext,newtxmath,microtype}
\RequirePackage{graphicx,booktabs,colortbl,float,placeins,etoolbox,authblk,adjustbox}
\RequirePackage[font=small,labelfont=bf,labelsep=period]{caption}
\RequirePackage[numbers,sort&compress,square]{natbib}
\RequirePackage{bibunits}
\RequirePackage[hidelinks]{hyperref}

% Retain a comfortable measure while accommodating the long kernel equations.
\AtBeginDocument{\newgeometry{textwidth=7.1in,textheight=9in,top=1in,
  headheight=14pt,headsep=20pt,footskip=30pt}}
\renewcommand{\shorttitle}{\small Neural density functional for polymer electrolytes}
\renewcommand{\Authfont}{\normalsize}
\renewcommand{\Affilfont}{\small\normalfont}
\setlength{\affilsep}{3pt}
\setlength{\bibsep}{2pt}
\renewcommand{\bibfont}{\small}
\setlength{\emergencystretch}{1em}
\raggedbottom

\newcommand{\tablenote}[1]{{\footnotesize\raggedright#1\par}}
\AtBeginEnvironment{tabular}{\sffamily\fontsize{7.5}{10}\selectfont}
\AtBeginEnvironment{tabular*}{\sffamily\fontsize{7.5}{10}\selectfont}
\newcommand{\preprinttables}{%
  \BeforeBeginEnvironment{tabular}{\begin{adjustbox}{max width=\linewidth}}%
  \AfterEndEnvironment{tabular}{\end{adjustbox}}%
}
