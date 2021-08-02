import * as cdk from '@aws-cdk/core';
import * as lambda from '@aws-cdk/aws-lambda';
import * as s3 from '@aws-cdk/aws-s3';
import * as iam from '@aws-cdk/aws-iam';
import {S3EventSource} from "@aws-cdk/aws-lambda-event-sources";


export class VehiclesLicensePlateDetectionAndRecognitionStack extends cdk.Stack {
  constructor(scope: cdk.Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    /**
     * User Parameters Configuration: Depends on User's Selection
     */
    const processFramesInterval = new cdk.CfnParameter(this, 'processFramesInterval', {
        description: 'Please configure the frame interval for license detection and recognition',
        type: 'String',
        default: '10',
        allowedValues: [
            '5',
            '10',
            '15',
            '20',
            '25',
            '30'
        ]
    });


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
     * S3 bucket provision: Create a bucket which is used to store all inference results
     */
    const inferenceResults = new s3.Bucket(
        this,
        'inferenceResults',
        {
            removalPolicy: cdk.RemovalPolicy.DESTROY,
            autoDeleteObjects: true,
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
                VideoAssetsS3BucketName: videosAsset.bucketName,
                InferenceResultsS3BucketName: inferenceResults.bucketName,
                FramesInterval: processFramesInterval.valueAsString,
            },
            timeout: cdk.Duration.minutes(15),
            role: role,
            memorySize: 10240,
        }
    );


    /**
    * Add S3 trigger event
    * Every time a .mp4 / .ts video clip is uploaded into S3 bucket, it triggers the lambda function to
    * extract key frames and perform license plate detection and recognition. Finally the results are
    * dumped into S3 bucket
    */
    frameExtractorWithLicensePlateDetectionAndRecognition.addEventSource(new S3EventSource(
        videosAsset,
        {
            events: [ s3.EventType.OBJECT_CREATED],
            filters: [ { suffix: '.mp4' } ]
    }));
    frameExtractorWithLicensePlateDetectionAndRecognition.addEventSource(new S3EventSource(
        videosAsset,
        {
            events: [ s3.EventType.OBJECT_CREATED],
            filters: [ { suffix: '.ts' } ]
    }));

  }
}
