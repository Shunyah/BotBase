#include <iostream>
#include <opencv2/opencv.hpp>
#include <ViZDoom.h>
#include <chrono>
#include <thread>
#include <opencv2/core/mat.hpp>

double total_reward = 0;

std::string path = "..\\vizdoom";
auto game = std::make_unique<vizdoom::DoomGame>();
auto screenBuff = cv::Mat(480, 640, CV_8UC3);
const unsigned int sleepTime = 1000 / vizdoom::DEFAULT_TICRATE;

void game_init() {
	game->setScreenResolution(vizdoom::RES_640X480);
	game->setLabelsBufferEnabled(true); // add this
	game->setWindowVisible(true);
	game->setRenderWeapon(true);
	game->setRenderHud(true);
}

void RunTask1(int episodes)
{
	try
	{
		game->loadConfig("../vizdoom/scenarios/task1.cfg"); 
		//game->setLabelsBufferEnabled(true);
		//game->setWindowVisible(true);
		//game->setRenderWeapon(true);
		game->init();
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}

	auto image = cv::Mat(480, 640, CV_8UC3);
	auto greyscale = cv::Mat(480, 640, CV_8UC1);

	std::vector<double> actions[4];

	actions[1] = { 1,0,0 };
	actions[2] = { 0,1,0 };
	actions[3] = { 0,0,1 };
	actions[4] = { 0,0,0 };

	cv::Mat clusters;
		for (auto i = 0; i < episodes; i++)
	{
		game->newEpisode();
		std::cout << "Episode #" << i + 1 << std::endl;

while (!game->isEpisodeFinished())
{
	const auto& gamestate = game->getState();
	std::memcpy(image.data, gamestate->screenBuffer->data(), gamestate->screenBuffer->size());

	cv::extractChannel(image, greyscale, 2);

cv:threshold(greyscale, greyscale, 175, 255, cv::THRESH_BINARY);

	//cv::imshow("Output Window", greyscale);

	//greyscale.convertTo(greyscale, CV_8UC3);

	std::vector<cv::Point2f> data;
	std::vector<cv::Point2f> centers(0);

	for (int x = 0; x < 640; x++)
	{
		for (int y = 0; y < 480; y++)
		{
			//if (int(image.at<cv::Vec3b>(y, x)[2]) > 130 && int(image.at<cv::Vec3b>(y, x)[0]) < 50)
			if ((int)greyscale.at<unsigned char>(x,y)==255)
			{
				data.push_back(cv::Point2f(x, y));
			}
		}
	}

	greyscale.convertTo(greyscale, CV_32F);

	//cv::Mat samples = greyscale.reshape(1, greyscale.total());

	cv::kmeans(data, 2, clusters, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0), 3, cv::KMEANS_RANDOM_CENTERS, centers);

	greyscale.convertTo(greyscale, CV_8UC3);

	//for (int i = 0; i < data.size(); i++)
	//{
		//cv::Point c = data[i];

		//cv::circle(image, c, 5, cv::Scalar(0, 0, 255), -1, 8);
		//cv::rectangle(image, cv::Rect(c.x - 25, c.y - 25, 50, 50), cv::Scalar(0, 0, 255));}

	//for (int i = 0; i < centers.size(); i++){
//		cv::circle(image, centers[i], 2, cv::Scalar(0, 255, 0));}

	imshow("Game", image);
	imshow("Greyscale", greyscale);
	cv::moveWindow("Game", 60, 20);
	cv::moveWindow("Greyscale", 710, 20);

	float y1 = centers[0].y;
	float y2 = centers[1].y;
	double eps = 21;
	if (y2 < y1) { // тогда y[0] демон 

		if (centers[0].x + 36 < centers[1].x) {
			game->makeAction(actions[2]); // лево
		}
		else if (centers[0].x - 30 > centers[1].x) {
			game->makeAction(actions[1]); // право
		}
		else {
			game->makeAction(actions[3]); // shoot
		}
	}
	else {

		if (centers[1].x + 35 < centers[0].x) {
			game->makeAction(actions[2]); //лево
		}
		else if (centers[1].x - 30 > centers[0].x) {
			game->makeAction(actions[1]); // право
		}
		else {
			game->makeAction(actions[3]); // shoot
		}
	}

	char c = cv::waitKey(sleepTime);

	//if 'ESC'
	if (c == 27) break;
}
std::cout << game->getTotalReward() << std::endl;
total_reward += game->getTotalReward();
		}
}


int main() {
	game->setViZDoomPath("../vizdoom/vizdoom");
	game->setDoomGamePath("../vizdoom/freedoom2.wad");

	cv::namedWindow("Output Window", cv::WINDOW_AUTOSIZE);

	auto episodes = 10;

	//------------------
	RunTask1(episodes);
	//------------------

	game->close();
}
