#merge data
import pandas as pd
import numpy as np

df_gpcr = pd.read_csv('NSG_gpcr_chemical.csv',index_col='ID')
df_tox21 = pd.read_csv('NSG_tox21_chemical.csv',index_col='CAS')
df1 = df_gpcr[['neuSmi','y']]
df2 = df_tox21[['neuSmi','y']]
df1['src'] = 'GPCR'
df2['src'] = 'TOX21'
data = pd.concat([df1,df2],axis=0)
#Please note: This study characterizes only the chemical space, hence all y values are set to zero. 
#The y values of datasets can be retained if you want to characterize the feature-response landscape of the dataset.
data['y'] = 0
len(data[data['y'] == 0])
data.to_csv('total_data_for_MML.csv')

# Please install the ADSAL package first; the NSG-related core code has been incorporated into ADSAL.
# ADSAL is already available online: https://test.pypi.org/project/adsal/

import adsal
from packaging import version

# Minimum required version for metAppDomain_gMark3
MIN_VERSION = "0.7.4"

if version.parse(adsal.__version__) < version.parse(MIN_VERSION):
    raise RuntimeError(
        f"The installed 'adsal' version ({adsal.__version__}) is too old.\n"
        f"Please upgrade to version {MIN_VERSION} or higher to use metAppDomain_gMark3.\n"
        f"Run: pip install --upgrade adsal"
    )

# Safe to import the new module
from adsal.metAppDomain_gMark3 import NSG, NSGVisualizer
import matplotlib.pyplot as plt

data = pd.read_csv('total_data_for_MML.csv')
data['dataset'] = 'lightgreen'
data.loc[(data.src == 'TOX21'),'dataset'] = 'orangered'
data['edge'] = '0.65'
data.loc[(data.src == 'TOX21'),'edge'] = '0.65'

#Initialize NSG
nsg = NSG(data, smiCol='neuSmi', yCol='y')
nsg.calcPairwiseSimilarityWithFp('MACCS_keys')
#nsg.calcPairwiseSimilarityWithFp('Morgan(bit)', radius=2, nBits=1024)
sCutoff = 0.85
#sCutoff = 0.55
stf = 15
#eps = 0.001
nsg.genNSG(sCutoff=sCutoff)
nsg.genGss()
nsg.calcCC()

dff = nsg.calcSoftLocalDiscontinuityScore2(sCutoff,stf)
uList = nsg.filterCCbySize(8)
#uList = nsg.filterCCwithNodes(['cid16759579','cid17757278','cid25246343'])
nsg.genGss(uList)
#NSG Visualization
nsgview = NSGVisualizer(nsg)
nsgview.nsg.df_train = nsgview.nsg.df_train.join(dff['softLD|{:s}|{:.2f}'.format(nsg.fpTypeStr,nsg.GsCutoff)])
nsgview.nsg.df_train = nsgview.nsg.df_train.join(data['src'])
nsgview.calcPos(prog='neato',picklePos=True,pickledPosPrefix='Focus-')
nsg.calcComm()
nsgview.nsg.df_train
nsgview.nsg.df_train.to_csv('Chemical_space_set.csv')

data['dataset'] = '#6DD0FA'
data.loc[(data.src == 'TOX21'),'dataset'] = '#FF9301'
data['edge'] = '0.65'
data.loc[(data.src == 'TOX21'),'edge'] = '0.65'

nsgview.render(
    'LD|{:s}|{:.2f}'.format(nsg.fpTypeStr,nsg.GsCutoff),
    'y',
    nodeEdgeColors= data.loc[uList,'dataset'],
    nodeshape = 'o',
    nodeEdgeWeights= data.loc[uList,'edge'],
    drawNodeLabels=False,
    nodesShown=None,
    font_size=8,
    markComm=True, # do not draw irrelavant communities
    minCommSize=1,
    commEdgeColor='gray',
    commAlpha=0.02,
    commEdgeWidth=0.5,
    commEdgePad=40,
    commEdgeInterpN=36,
    ratioAdjust=None,
    figsizeTup=(12.0, 8.0),
    legendWidth=0.7,
    drawEdges=True,
    edgeAlpha=0.25,
    vmin=0,
    vmax=3.0,
    isContinuousValue=False,
    nLegend_activity=1,
    leftPadRatio=0.1,
    rightPadRatio=0.1,
    bottomPadRatio=0.1,
    topPadRatio=0.1,
    annotateNodePos={},
    annotatePosStyle='offset points',
    sizeBase=0.30,
    sizeScale=20,
    cmapName='binary',
    nLegend_LD=4,
    bboxTup=(0.25, 0.5, 0.25, 0.8),
    showLegend=True,
    savePng=True,
    PngNamePrefix='Focus_',
    groupsToMark=[]
)



