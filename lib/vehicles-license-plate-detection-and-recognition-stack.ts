import * as cdk from '@aws-cdk/core';
import * as lambda from '@aws-cdk/aws-lambda';
import * as s3 from '@aws-cdk/aws-s3';
import * as iam from '@aws-cdk/aws-iam';
import * as dynamodb from '@aws-cdk/aws-dynamodb';
import {S3EventSource} from "@aws-cdk/aws-lambda-event-sources";


export class VehiclesLicensePlateDetectionAndRecognitionStack extends cdk.Stack {
  constructor(scope: cdk.Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    /**
     * S3 bucket provision: Create a bucket which is used to store all video clips
     * When users upload video clips into this bucket, it triggers the lambda function
     * to perform vehicle detection and recognition
     */
    const videosAsset = new s3.Bucket(
        this,
        'videosAsset',
        {
            removalPolicy: cdk.RemovalPolicy.DESTROY,
            autoDeleteObjects: true,
        }
    );


    /**
    * Dynamodb Provision: it is used to store the inference results
    */
    const licensePlateInfoTable = new dynamodb.Table(
        this,
        'licensePlateInfoTable',
        {
            partitionKey: {
                name: 'event',
                type: dynamodb.AttributeType.STRING
            },
            billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
            removalPolicy: cdk.RemovalPolicy.DESTROY,
        }
    );


    /**
    * Lambda functions allow specifying their handlers within docker images. The docker
    * image can be an image from ECR or a local asset that the CDK will package and load
    * into ECR. The following `DockerImageFunction` construct uses a local folder with a
    * Dockerfile as the asset that will be used as the function handler.
    */
    const role = new iam.Role(
        this,
        'lambdaExecuteRole',
        {
            assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
            managedPolicies: [
                iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonS3FullAccess'),
                iam.ManagedPolicy.fromAwsManagedPolicyName('CloudWatchLogsFullAccess'),
            ]
    });

    const frameExtractorWithLicensePlateDetectionAndRecognition = new lambda.DockerImageFunction(
        this,
        'frameExtractorWithLicensePlateDetectionAndRecognition',
        {
            code: lambda.DockerImageCode.fromImageAsset('./lambda/'),
            environment: {
                S3BucketName: videosAsset.bucketName,
                DynamoDBName: licensePlateInfoTable.tableName,
                DynamoDBPrimaryKey: 'event'
            },
            timeout: cdk.Duration.minutes(15),
            role: role,
            memorySize: 10240,
        }
    );
    // assign dynamodb permissions for lambda functions
    licensePlateInfoTable.grantWriteData(frameExtractorWithLicensePlateDetectionAndRecognition);


    /**
    * Add S3 trigger event
    * Every time a .mp4 / .ts video clip is uploaded into S3 bucket, it triggers the lambda function to
    * extract key frames and perform license plate detection and recognition. Finally the results are
    * dumped into DynamoDB
    */
    frameExtractorWithLicensePlateDetectionAndRecognition.addEventSource(new S3EventSource(
        videosAsset,
        {
            events: [ s3.EventType.OBJECT_CREATED],
            filters: [
                { suffix: '.mp4' },
                { suffix: '.ts' }
            ]
        }
    ));

  }
}
