import csv


def writCSV(path, prediction):
    with open(path, 'w', newline='') as csvFile:
                fileNames = ['id', 'happiness']
                writer = csv.DictWriter(csvFile, fieldnames=fileNames)
                writer.writeheader()
                for i in range(len(prediction)):
                    dictionary = {'id': i+8001, 'happiness': prediction[i]}
                    writer.writerow(dictionary)
