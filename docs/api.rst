.. _api:

API Reference
=============

.. module:: pypdf_table_extraction

Main Interface
--------------
.. autofunction:: pypdf_table_extraction.read_pdf

Lower-Level Classes
-------------------

.. autoclass:: pypdf_table_extraction.handlers.PDFHandler
   :inherited-members:

.. autoclass:: pypdf_table_extraction.parsers.Stream
   :inherited-members:

.. autoclass:: pypdf_table_extraction.parsers.Lattice
   :inherited-members:

.. autoclass:: pypdf_table_extraction.parsers.Network
   :inherited-members:

.. autoclass:: pypdf_table_extraction.parsers.Hybrid
   :inherited-members:

Lower-Lower-Level Classes
-------------------------

.. autoclass:: pypdf_table_extraction.core.TableList
   :inherited-members:

.. autoclass:: pypdf_table_extraction.core.Table
   :inherited-members:

.. autoclass:: pypdf_table_extraction.core.Cell

Plotting
--------

.. autofunction:: pypdf_table_extraction.plot

.. autoclass:: pypdf_table_extraction.plotting.PlotMethods
   :inherited-members:
