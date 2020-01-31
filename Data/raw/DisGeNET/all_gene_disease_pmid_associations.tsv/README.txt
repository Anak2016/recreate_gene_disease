**********************************
 DisGeNET, a discovery platform for human diseases and their genes
**********************************

DisGeNET is a discovery platform integrating information on gene-disease associations from several public data sources and the literature.
The DisGeNET data is made available under the Attribution-NonCommercial-ShareAlike 4.0 International License.
For more information, see the Legal Notices page (http://www.disgenet.org/ds/DisGeNET/html/legal.html).
The files in the current directory contain the data corresponding to the latest release (version 6.0, January 2019).
The information is separated by tab. When field are concatenated, they are separated by ";"

curated_gene_disease_associations.tsv.gz 		=> Gene-Disease associations from UniProt, CGI, ClinGen, Genomics England Panel App, PsyGeNET, Orphanet, the HPO, and CTD (human data)
befree_gene_disease_associations.tsv.gz 		=> Gene-Disease associations obtained using BeFree System
all_gene_disease_associations.tsv.gz 			=> All Gene-Disease associations in DisGeNET

The columns in the files are:
geneId 		-> NCBI Entrez Gene Identifier
geneSymbol	-> Official Gene Symbol
DSI		-> The Disease Specificity Index for the gene
DPI		-> The Disease Pleiotropy Index for the gene
diseaseId 	-> UMLS concept unique identifier
diseaseName 	-> Name of the disease
diseaseType  	-> The DisGeNET disease type: disease, phenotype and group
diseaseClass	-> The MeSH disease class(es)
diseaseSemanticType	-> The UMLS Semantic Type(s) of the disease
score		-> DisGENET score for the Gene-Disease association
EI		-> The Evidence Index for the Gene-Disease association
YearInitial	-> First time that the Gene-Disease association was reported
YearFinal	-> Last time that the Gene-Disease association was reported
NofPmids	-> Total number of publications reporting the Gene-Disease association
NofSnps		-> Total number of SNPs associated to the Gene-Disease association
source		-> Original source reporting the Gene-Disease association

curated_variant_disease_associations.tsv.gz 		=> Variant-Disease associations from UniProt, ClinVar, GWASdb and the GWAS Catalog
befree_variant_disease_associations.tsv.gz 		=> Variant-Disease associations obtained using BeFree System
all_variant_disease_associations.tsv.gz 		=> All Variant-Disease associations in DisGeNET

The columns in the files are:
snpId 		-> dbSNP variant Identifier
chromosome	-> Chromosome of the variant
position	-> Position in chromosome
DSI		-> The Disease Specificity Index for the variant
DPI		-> The Disease Pleiotropy Index for the variant
diseaseId 	-> UMLS concept unique identifier
diseaseName 	-> Name of the disease
diseaseType  	-> The DisGeNET disease type: disease, phenotype and group
diseaseClass	-> The MeSH disease class(es)
diseaseSemanticType	-> The UMLS Semantic Type(s) of the disease
score		-> DisGENET score for the Variant-Disease association
EI		-> The Evidence Index for the Variant-Disease association
YearInitial	-> First time that the Variant-Disease association was reported
YearFinal	-> Last time that the Variant-Disease association was reported
NofPmids	-> Total number of publications reporting the Variant-Disease association
source		-> Original source reporting the Variant-Disease association

all_gene_disease_pmid_associations.tsv.gz		=> All Gene-Disease-PMID associations in DisGeNET

The columns in the files are:
geneId 		-> NCBI Entrez Gene Identifier
geneSymbol	-> Official Gene Symbol
DSI		-> The Disease Specificity Index for the gene
DPI		-> The Disease Pleiotropy Index for the gene
diseaseId 	-> UMLS concept unique identifier
diseaseName 	-> Name of the disease
diseaseType  	-> The DisGeNET disease type: disease, phenotype and group
diseaseClass	-> The MeSH disease class(es)
diseaseSemanticType	-> The UMLS Semantic Type(s) of the disease
score		-> DisGENET score for the Gene-Disease association
EI		-> The Evidence Index for the Gene-Disease association
YearInitial	-> First time that the Gene-Disease association was reported
YearFinal	-> Last time that the Gene-Disease association was reported
pmid		-> Publication reporting the Gene-Disease association
source		-> Original source reporting the Gene-Disease association

all_variant_disease_pmid_associations.tsv.gz 		=> All Variant-Disease-PMID associations in DisGeNET

The columns in the files are:
snpId 		-> dbSNP variant Identifier
chromosome	-> Chromosome of the variant
position	-> Position in chromosome
DSI	-> The Disease Specificity Index for the variant
DPI	-> The Disease Pleiotropy Index for the variant
diseaseId 	-> UMLS concept unique identifier
diseaseName 	-> Name of the disease
diseaseType 	-> disease, phenotype, or group
diseaseType  	-> The DisGeNET disease type: disease, phenotype and group
diseaseClass	-> The MeSH disease class(es)
score		-> DisGENET score for the Variant-Disease association
YearInitial	-> First time that the Variant-Disease association was reported
YearFinal	-> Last time that the Variant-Disease association was reported
pmid		-> Publication reporting the Variant-Disease association
source		-> Original source reporting the Variant-Disease association


disease_mappings.tsv.gz				=> Mappings from UMLS concept unique identifier to disease vocabularies: DO, EFO, HPO, ICD9CM, MSH, NCI, OMIM, and ORDO

variant_to_gene_mappings.tsv.gz 		=> Variant mapped to their corresponding genes, according to dbSNP.

The columns in the files are:
snpId 		-> dbSNP variant Identifier
geneId		-> NCBI Entrez Gene Identifier
geneSymbol		-> Official Gene Symbol

Disclaimer

Except where expressly provided otherwise, the site, and all content, materials, information, software, products and services provided on the site, are provided on an "as is" and "as available" basis. The IBI group expressly disclaims all warranties of any kind, whether express or implied, including, but not limited to, the implied warranties of merchantability, fitness for a particular purpose and non-infringement. The IBI group makes no warranty that:

    a. the site will meet your requirements

    b. the site will be available on an uninterrupted, timely, secure, or error-free basis (though IBI will undertake best-efforts to ensure continual uptime and availability of its content)

    c. the results that may be obtained from the use of the site or any services offered through the site will be accurate or reliable

    d. the quality of any products, services, information, or other material obtained by you through the site will meet your expectations

Any content, materials, information or software downloaded or otherwise obtained through the use of the site is done at your own discretion and risk. The IBI group shall have no responsibility for any damage to your computer system or loss of data that results from the download of any content, materials, information or software. The IBI group reserves the right to make changes or updates to the site at any time without notice.

If you have any further questions, please email us at support@disgenet.org

last modified: May, 2019
SAVE TO CACHER