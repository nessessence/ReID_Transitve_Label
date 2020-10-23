
import concurrent.futures
from itertools import repeat
import boto3 

def upload_file(s3_client,bucket,file_path,object_name=None):
    """Upload a file to an S3 bucket
    :param file_path: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """
    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = "/".join(file_path.split("/")[-2:]) # fdir/img.jpg
    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_path, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

def upload_multiple_files(file_paths,bucket,object_names=None):
    if object_names is None:
        object_names = file_paths
    s3_client = boto3.client('s3')
    print("start uploading")
    with concurrent.futures.ThreadPoolExecutor() as executor:
         results = list(executor.map(upload_file,repeat(s3_client),repeat(bucket),file_paths))
    Failed_upload = [ i for i,x in  enumerate(file_paths) if not x]
    if len(Failed_upload) > 0  :  print(f"Warning: There are {len(Failed_upload)} files that fail to upload")
    else: print("complete uploading all files")
    return Failed_upload

# print(list(upload_multiple_files(["nasa.png","twins.jpg"],'ness-sds-bucket')))



import pymongo
from pymongo import MongoClient
from pymongo import UpdateOne
from pymongo.errors import BulkWriteError


def upload_to_db(db_config,storage,k,rank_mat,img_paths):
    print(f"start initializeing db: {db_config['URI']} ")
    print(f"using {db_config['collection']}")
    connection = MongoClient(db_config["URI"])
    database = connection[db_config["name"]]
    collection = database[db_config["collection"]]

    #fdir_filenames = [ final_dir+"/"+file_name for path in img_paths for final_dir,file_name  in [path.split("/")[-2:]] ] # final_directory/filename.jpg 
    # init doc values
    fdir_filenames = []; dates = []; camIds = [] # {camId}/{20xx-xx-xx} yyyy.jpg
    for path in img_paths: 
        for final_dir,file_name  in [path.split("/")[-2:]]:
            fdir_filenames.append(final_dir+"/"+file_name)
            dates.append(file_name.split(" ")[0])
            camIds.append(final_dir)
    docs = [{
        "key":  fdir_filenames[i],
        "storage": storage,
        "date": dates[i],
        "camId": camIds[i],
        "lstatus": 0,
        "lid": {},
        "k": k,
        "rank_list": [fdir_filenames[j] for j in rank_order if j!=i] 
        }  for i,rank_order in enumerate(rank_mat)] 
    # {} --> will be an empty object in MongoDB
    requests = [  UpdateOne({ "key": doc["key"]}, { "$set": doc }, upsert=True) for doc in docs ]
    try:
        collection.bulk_write(requests)
    except BulkWriteError as bwe:
        print(bwe.details)
    print("finished uploading to database")

    





