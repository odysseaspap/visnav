#include "slamBase.h"

PointCloud::Ptr image2PointCloud( cv::Mat& rgb, cv::Mat& depth, CAMERA_INTRINSIC_PARAMETERS& camera )
{
    PointCloud::Ptr cloud ( new PointCloud );

    for (int m = 0; m < depth.rows; m+=2)
        for (int n=0; n < depth.cols; n+=2)
        {
            ushort d = depth.ptr<ushort>(m)[n];
            if (d == 0)
                continue;
            PointT p;

            p.z = double(d) / camera.scale;
            p.x = (n - camera.cx) * p.z / camera.fx;
            p.y = (m - camera.cy) * p.z / camera.fy;

            p.b = rgb.ptr<uchar>(m)[n*3];
            p.g = rgb.ptr<uchar>(m)[n*3+1];
            p.r = rgb.ptr<uchar>(m)[n*3+2];

            cloud->points.push_back( p );
        }
    cloud->height = 1;
    cloud->width = cloud->points.size();
    cloud->is_dense = false;

    return cloud;
}

cv::Point3f point2dTo3d( cv::Point3f& point, CAMERA_INTRINSIC_PARAMETERS& camera )
{
    cv::Point3f p;
    p.z = double( point.z ) / camera.scale;
    p.x = ( point.x - camera.cx) * p.z / camera.fx;
    p.y = ( point.y - camera.cy) * p.z / camera.fy;
    return p;
}

void computeKeyPointsAndDesp( FRAME& frame, string detector, string descriptor )
{
    cv::Ptr<cv::FeatureDetector> _detector;
    cv::Ptr<cv::DescriptorExtractor> _descriptor;

    _detector = cv::ORB::create();
    _descriptor = cv::ORB::create();

    if (!_detector || !_descriptor)
    {
        cerr<<"Unknown detector or discriptor type !"<<detector<<","<<descriptor<<endl;
        return;
    }

    _detector->detect( frame.rgb, frame.kp );
    _descriptor->compute( frame.rgb, frame.kp, frame.desp );

    //cv::Mat imgShow;
    //cv::drawKeypoints( frame.rgb, frame.kp, imgShow, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    //cv::imshow( "keypoints", imgShow );

    return;
}

RESULT_OF_PNP estimateMotion( FRAME& frame1, FRAME& frame2, CAMERA_INTRINSIC_PARAMETERS& camera )
{


    static ParameterReader pd;
    vector< cv::DMatch > matches;
    cv::BFMatcher matcher;
    matcher.match( frame1.desp, frame2.desp, matches );

    RESULT_OF_PNP result;
    vector< cv::DMatch > goodMatches;
    double minDis = 9999;
    double good_match_threshold = atof( pd.getData( "good_match_threshold" ).c_str() );
    for ( size_t i=0; i<matches.size(); i++ )
    {
        if ( matches[i].distance < minDis )
            minDis = matches[i].distance;
    }

    cout << "minimum distance" << minDis << endl;

    if ( minDis < 10 )
        minDis = 10;


    for ( size_t i=0; i<matches.size(); i++ )
    {
        if (matches[i].distance < good_match_threshold*minDis)//max(minDis*2.0,30.0))
            goodMatches.push_back( matches[i] );
    }


    if (goodMatches.size() <= 5)
    {
        result.inliers = -1;
        return result;
    }

      //-- Draw only "good" matches
    cv::Mat img_matches;
    cv::drawMatches( frame1.rgb, frame1.kp, frame2.rgb, frame1.kp,
               goodMatches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
               std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    //-- Show detected matches
    imshow( "Good Matches", img_matches );
    //cv::waitKey(0);

    vector<cv::Point3f> pts_obj;
    vector< cv::Point2f > pts_img;

    for (size_t i=0; i<goodMatches.size(); i++)
    {
        cv::Point2f p = frame1.kp[goodMatches[i].queryIdx].pt;
        ushort d = frame1.depth.ptr<ushort>( int(p.y) )[ int(p.x) ];
        if (d == 0)
            continue;
        pts_img.push_back( cv::Point2f( frame2.kp[goodMatches[i].trainIdx].pt ) );

        cv::Point3f pt ( p.x, p.y, d );
        cv::Point3f pd = point2dTo3d( pt, camera );
        pts_obj.push_back( pd );
    }

    if (pts_obj.size() ==0 || pts_img.size()==0)
    {
        result.inliers = -1;
        return result;
    }

    double camera_matrix_data[3][3] = {
        {camera.fx, 0, camera.cx},
        {0, camera.fy, camera.cy},
        {0, 0, 1}
    };


    cv::Mat cameraMatrix( 3, 3, CV_64FC1, camera_matrix_data );
    cv::Mat rvec, tvec, inliers;
    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64FC1);

    cv::solvePnPRansac( pts_obj, pts_img, cameraMatrix, distCoeffs, rvec, tvec, false, 100, 8.0, 0.99, inliers );

    result.rvec = rvec;
    result.tvec = tvec;
    result.inliers = inliers.rows;

    return result;
}


// cvMat2Eigen
Eigen::Isometry3d cvMat2Eigen( cv::Mat& rvec, cv::Mat& tvec )
{
    cv::Mat R;
    cv::Rodrigues( rvec, R );
    Eigen::Matrix3d r;
    for ( int i=0; i<3; i++ )
        for ( int j=0; j<3; j++ )
            r(i,j) = R.at<double>(i,j);

    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();

    Eigen::AngleAxisd angle(r);
    T = angle;
    T(0,3) = tvec.at<double>(0,0);
    T(1,3) = tvec.at<double>(1,0);
    T(2,3) = tvec.at<double>(2,0);
    return T;
}

PointCloud::Ptr joinPointCloud( PointCloud::Ptr original, FRAME& newFrame, Eigen::Isometry3d T, CAMERA_INTRINSIC_PARAMETERS& camera )
{
    PointCloud::Ptr newCloud = image2PointCloud( newFrame.rgb, newFrame.depth, camera );

    PointCloud::Ptr output (new PointCloud());
    pcl::transformPointCloud( *original, *output, T.matrix() );
    *newCloud += *output;

    static pcl::VoxelGrid<PointT> voxel;
    static ParameterReader pd;
    double gridsize = atof( pd.getData("voxel_grid").c_str() );
    voxel.setLeafSize( gridsize, gridsize, gridsize );
    voxel.setInputCloud( newCloud );
    PointCloud::Ptr tmp( new PointCloud() );
    voxel.filter( *tmp );
    return tmp;
}
