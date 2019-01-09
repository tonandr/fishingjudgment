import fj
from py4j.java_gateway import *
gateway = JavaGateway(auto_field=True, auto_convert=True)
java_import(gateway.jvm, "org.apache.spark.SparkConf")
conf = gateway.jvm.SparkConf().setMaster("local[*]").setAppName("SparkNN")
sc = gateway.entry_point.getSparkContext(conf)
fjm = fj.FishingJudgment(gateway, sc)
trainingCSVFilePath = '/Users/gutoo/topcoder/fishing_judging/data' + '/training_data.csv'
testCSVFilePath = '/Users/gutoo/topcoder/fishing_judging/data' + 'testing_data.csv'
numLayers = 3; numActs = [11, 10, 1]
fjm.train(trainingCSVFilePath, numLayers, numActs, 0.0, loadFlag=True)