# A Comparative Study of Diffusion Models and Long Short-Term Memory Networks throught the Reaction–Diffusion Equation

This is a Research Project led in the Mathematical Informatics laboratory at Nara Institute of Science and Technology.

## Abstract 
Understanding how complex patterns emerge from simple rules is a central question in both biology and artificial intelligence. In biological systems, Alan Turing’s Reaction–diffusion (RD) model provides an explanation for spontaneous pattern formation. It describes how interacting chemical substances, an activator and an inhibitor, diffuse through space and react locally, generating stable spatial patterns such as stripes, spots, or waves. This interaction between local activation and global inhibition has since been recognized as a general mechanism for self-organization in living systems. In parallel, modern machine learning models have revealed similar dynamics in artificial systems. Long Short-Term Memory (LSTM) networks are a class of recurrent neural architectures designed to model temporal sequences and retain information over long time series data. They achieve this by regulating how past states influence future predictions, allowing them to capture both short-term and long-term dependencies in dynamic data. In a similar perspective, Denoising Diffusion Probabilistic Models (DDPMs) construct structured data from random noise through an iterative denoising process that progressively reverses a stochastic Gaussian noise diffusion. Although all these frameworks are used in different disciplines, they share a common mathematical foundation: the dynamic balance between growth and decay, reaction and diffusion. Investigating these parallels provides an understanding toward pattern formation, stability, and feedback control in both natural and artificial systems. However, the literature lacks a formal mathematical framework linking these two machine learning models. Existing studies rarely examine how the internal dynamics of artificial neural networks can be interpreted using concepts from biological reaction–diffusion system. This absence leaves unexplored a potentially rich analogy between biological self-organization and artificial memory regulation.

In this work, we review the similarities between DDPMs and LSTM networks throught the Reaction-diffusion equation. Morphogen concentrations variate depending on a diffusion and a reaction component. LSTM networks cell-state iterative equation is analogous to this mechanism but in a discrete framework due to the its squashing functions. Likewise, DDPMs viewed as continuous-time stochastic differential equations highlights a striking analogy thought the presence of both drift and diffusion terms. Hence, a general mathematical equation can emerge that unifies the three models main output. The concept of short-range positive feedback and long-range negative feedback in the RD model emphasize the parallel as well. In LSTM networks, the input gate, which amplifies and stores relevant signals, and the forget gate, which dissipates or suppresses irrelevant information, respectively act as the activator and the inhibitor in the RD model. In DDPMs, the denoising network provides local positive feedback, restoring structure lost through noise, while the diffusion process applies global negative feedback, spreading and stabilizing information

## Objectives

In this git repository, we aim to reproduce turing patterns using a LSTM RNN and DDPM and then compare the result in stability and complexity. This first step is to implement the Reaction-Diffusion equation system.

## Gray-Scott Model

$$
\begin{align}
    \frac{\partial U}{\partial t} &= D_U \nabla^2 U - UV^2 + F(1-U)\\
    \frac{\partial V}{\partial t} &= D_V \nabla^2 V + UV^2 - (F+k) V
\end{align}
$$

Where the terms containing $D_U$ and $D_V$ are the diffusion terms and the rest are reaction terms.