import { expect as expectCDK, matchTemplate, MatchStyle } from '@aws-cdk/assert';
import * as cdk from '@aws-cdk/core';
import * as VehiclesLicensePlateDetectionAndRecognition from '../lib/vehicles-license-plate-detection-and-recognition-stack';

test('Empty Stack', () => {
    const app = new cdk.App();
    // WHEN
    const stack = new VehiclesLicensePlateDetectionAndRecognition.VehiclesLicensePlateDetectionAndRecognitionStack(app, 'MyTestStack');
    // THEN
    expectCDK(stack).to(matchTemplate({
      "Resources": {}
    }, MatchStyle.EXACT))
});
