#
# Routine to download files from web for WebFOCUS reporting
#


# import Python libraries
import sys
import pandas as pd
import time
import csv

def describe_dataset(dataset):
    print('\nDataset information:')
    print(dataset.info())
    print(dataset.dtypes)
    print('\nDataset shape (rows,columns):', dataset.shape)
    print('\nDataset first 10 records:')
    print(dataset.head(10))
    print('\nDataset last 10 records:')
    print(dataset.tail(10))
    #print('\nDataset statuscurrent distribution:')
    #print(dataset.groupby('statuscurrent').size())
    #print('\nDataset statistics by statuscurrent:')
    #print(dataset.describe())


# copy the web file to a WFRS folder
def write_webfocus_copy(dataset, filedescription, output_data):
    print('Writing %s file to WebFOCUS Reporting Server: %s...' % (filedescription, output_data))
    dataset.to_csv(output_data, encoding='utf-8', sep=",", quoting=csv.QUOTE_NONNUMERIC, index=False)


# generate WebFOCUS synonym for the web data
def write_webfocus_synonym(dataset, filedescription, output_data, output_synonym):
    # figure out the lengths of the strings in the datafile

    metadatalengths = []
    for column in dataset:
        #print('debug: on column %s with dtype %s' % (column, dataset[column].dtype))
        if dataset[column].dtype == 'object':
            maxlength = dataset[column].astype(str).map(len).max()
            metadatalengths.append(maxlength)
            #print('Max length of column %s: %s' % (column, dataset[column].map(len).max()))
        else:
            metadatalengths.append(0)


    # output file for synonym
    print('Creating WebFOCUS synonym for %s: %s...' % (filedescription, output_synonym))
    mfdfile = open(output_synonym, 'w')

    mfdfile.write('$\n')
    mfdfile.write('$ %s synonym generated %s\n' % (filedescription, time.strftime('at %X on %m %B %Y %Z')))
    mfdfile.write('$\n')
    mfdfile.write('FILENAME=%s,SUFFIX=COMT,\n' % (filedescription))
    mfdfile.write('  DATASET=\'%s\',$\n' % (output_data))
    mfdfile.write('  SEGNAME=onlyseg,SEGTYPE=S0,$\n')
    counter = 0
    for colname, format in dataset.dtypes.iteritems():
        wfname = colname.replace(" ", "_")
        wftitle = colname.replace(" ", ",")
        if format == 'object':
            wfformat = "A%s" % (metadatalengths[counter])
        elif format == 'int64':
            wfformat = 'I4'
        elif format == 'float64':
            wfformat = 'D12.2'
        else:
            wfformat = 'WIP'

        mfdfile.write('    FIELDNAME=%s,ALIAS=%s,ACTUAL=%s,USAGE=%s,TITLE=\'%s\',$\n' % (wfname, wfname, wfformat, wfformat, wftitle))
        counter = counter + 1

    mfdfile.close()


# do the option for downloading Winter Olympics data from the web
def run_winter_olympics(filedescription, output_data, output_synonym):
    # read the Winter Olympics medal CSV file from the website
    #url = 'http://winterolympicsmedals.com/medals.csv'
    print('Downloading %s file from the web...' % (filedescription))
    url = 'http://www.kencura.com/data/winterolympicsmedals.csv'
    dataset = pd.read_csv(url)

    write_webfocus_copy(dataset, filedescription, output_data)
    write_webfocus_synonym(dataset, filedescription, output_data, output_synonym)




# do the option for downloading Winter Olympics data from the web
def run_cinci_data(filedescription, output_data, output_synonym, datacode):
    from sodapy import Socrata

    client = Socrata("data.cincinnati-oh.gov", None)
    results = client.get(datacode, limit=2000)

    # read the Cincinnati building permits CSV file from the website
    #url = 'http://https://data.cincinnati-oh.gov/resource/emnx-rw6d.csv'
    print('Downloading %s file from the web...' % (filedescription))
    #url = 'http://www.kencura.com/data/cincibuildingpermits.csv'
    #dataset = pd.read_csv(url)
    dataset = pd.DataFrame.from_records(results)

    # debug
    describe_dataset(dataset)

    write_webfocus_copy(dataset, filedescription, output_data)
    write_webfocus_synonym(dataset, filedescription, output_data, output_synonym)



# validate the parameters
def validate_arguments():
    print ('Running Python script named %s' % (sys.argv[0]))
    if len(sys.argv) > 1:
        print('\t#Arguments= ', len(sys.argv))
        print('\tArguments= ', str(sys.argv))
        count = 0
        while (count < len(sys.argv)):
            print ('\tArgument#%d is %s' % (count, sys.argv[count]))
            count = count + 1

        print()
    else:
        print ('No program arguments were provided.')
        exit()


# main function
def main():
    validate_arguments()
    wffolder = 'c:\\ibi\\apps\\webdownloads\\'
    filedescription = sys.argv[1]
    output_data = wffolder + filedescription + '.dat'
    output_synonym = wffolder + filedescription + '.mas'

    if sys.argv[1] == 'WinterOlympics':
        run_winter_olympics(filedescription, output_data, output_synonym)
    elif sys.argv[1] == 'CinciBuildingPermits':
        run_cinci_data(filedescription, output_data, output_synonym, "emnx-rw6d")
    elif sys.argv[1] == 'CinciHealthInspections':
        run_cinci_data(filedescription, output_data, output_synonym, "2c8u-zmu9")
    elif sys.argv[1] == 'CinciPoliceComplaints':
        run_cinci_data(filedescription, output_data, output_synonym, "5tnh-jksf")
    else:
        print('No download arguments were provided.')


# call main function
main()

