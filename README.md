# GCTSAAS

This repository contains the initial scaffolding for the **Grounded Coherence Theory** SaaS platform. The project aims to deliver exportable programs for industry customers to assess and improve organizational coherence.

## Overview

The `src/components/GCTArchitecture.tsx` file contains a React component that visualizes the proposed architecture of the system. Each component of the SaaS stack can be clicked to reveal implementation details and key considerations.

Future development will expand on this foundation to provide a fully functional platform.
## Real-Time Coherence Tracking

The `src/analytics/real_time_coherence.py` module contains a reference
implementation of the derivative calculations and streaming utilities
described in the technical architecture. Key improvements include:

- Derivatives respect the actual time spacing of measurements to avoid
  unit inconsistencies.
- The Savitzky--Golay window automatically adjusts to an odd number of
  samples and exposes polynomial order and smoothing settings via
  `DerivativeConfig`.
- Asynchronous Kafka consumers are implemented with `aiokafka` so
  derivative processing does not block the event loop.

This module is a starting point for the real-time coherence tracking
layer and will evolve alongside the rest of the platform.
