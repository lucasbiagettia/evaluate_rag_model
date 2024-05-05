from evaluate import load
import pandas as pd
import numpy as np

def compute_bleu(prediction, reference):
    bleu_metric = load("sacrebleu")
    bleu_metric.add(prediction=prediction, reference=reference)
    results = bleu_metric.compute(smooth_method="floor", smooth_value=0)
    results["precisions"] = [np.round(p, 2) for p in results["precisions"]]
    df = pd.DataFrame.from_dict(results, orient="index", columns=["Value"])
    return df



def compute_rouge(prediction, reference):
    rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    rouge_metric = load("rouge")
    rouge_metric.add(prediction=prediction, reference=reference)
    score = rouge_metric.compute()
    average_score = sum(score.values()) / len(score)    
    rouge_dict = dict((rn, score[rn]) for rn in rouge_names)
    df = pd.DataFrame.from_dict(rouge_dict, orient="index", columns=["Value"])
    print(df)
    return average_score


reference = [
    "Astor Pantaleón Piazzolla (Mar del Plata, 11 de marzo de 1921 – Buenos Aires, 4 de julio de 1992) fue un bandoneonista y compositor argentino considerado uno de los músicos más importantes del siglo xx4​5​ y uno de los compositores más importantes de tango en todo el mundo.6​",
    "Sus obras revolucionaron el tango tradicional en un nuevo estilo denominado nuevo tango o tango de vanguardia, incorporando elementos del jazz y la música clásica. Bandoneonista virtuoso, solía interpretar sus propias composiciones con una variedad de conjuntos. En 1992, el crítico de música estadounidense Stephen Holden describió a Piazzolla como «el compositor de música de tango más importante del mundo».7​",
    "Nació en Mar del Plata, pero desde muy joven se crio en la metrópolis de Nueva York, donde su padre le obsequió un bandoneón, el cual comenzó a tocar desde muy temprana edad. Tomó clases con Alberto Ginastera y ganó el Concurso Fabien Sevitzky, con el cual pudo financiarse un viaje a Europa para estudiar armonía, música clásica y contemporánea con la compositora y directora de orquesta francesa Nadia Boulanger. En su juventud tocó y realizó arreglos orquestales para el bandoneonista, compositor y director Aníbal Troilo. Cuando comenzó a hacer innovaciones en el tango en lo que respecta a ritmo, timbre y armonía, fue muy criticado por los tangueros de la «guardia vieja», ortodoxos en cuanto a ritmo, melodía y orquestación.",
    "Cuando en los años 1950 y 1960 los tangueros tradicionales —que lo consideraban «el asesino del tango»— decretaron que sus composiciones no eran tango, Piazzolla respondió con una nueva definición: «Es música contemporánea de Buenos Aires».8​9​ A pesar de esto, en Argentina las estaciones radiodifusoras no difundían sus obras y los comentaristas seguían atacando su arte. Durante años, los tangueros y críticos musicales lo consideraron un «esnob» irrespetuoso que componía 'música híbrida', con exabruptos de armonía disonante."
]

prediction = '''Astor Piazzolla fue un bandoneonista y compositor argentino, considerado uno de los músicos más importantes del siglo XX y uno de los compositores de tango más importantes del mundo. Revolucionó el tango tradicional con su "nuevo tango" o "tango de vanguardia", incorporando elementos del jazz y la música clásica.

Piazzolla nació en Mar del Plata, pero se crió en Nueva York. Estudió con Alberto Ginastera y Nadia Boulanger. Sus innovaciones en el tango le valieron críticas de los tangueros tradicionales, quienes lo consideraban un "asesino del tango". Piazzolla respondió definiendo su música como "contemporánea de Buenos Aires". A pesar de las críticas, Piazzolla es reconocido como un ícono del tango y uno de los músicos más influyentes del siglo XX.'''

print(compute_rouge(prediction, reference))