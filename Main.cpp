
#include <iostream>
#include <opencv2/opencv.hpp>
#include <ViZDoom.h>
#include <chrono>
#include <thread>
#include <opencv2/core/mat.hpp>

double total_reward = 0;

using namespace vizdoom;
using namespace cv;
using namespace std;

std::string path = "E:\\practice\\vizdoom";
auto game = std::make_unique <vizdoom::DoomGame>();
auto screenBuff = cv::Mat(480, 640, CV_8UC3);
const unsigned int sleepTime = 1000 / vizdoom::DEFAULT_TICRATE;

const std::vector<double> actions[3] = {
	{ 1, 0, 0 }, // left
	{ 0, 1, 0 }, // right
	{ 0, 0, 1 }, // shoot
	//{ 0, 0, 0, 1 }, // shoot
};

//void find_demon_and_kill(GameStatePtr state) {
//	double eps = 10; // monster's width
//	if (state->labels[0].objectPositionY - eps > state->labels[1].objectPositionY) {
//		game->makeAction(actions[0]); //left
//	}
//	else if (state->labels[0].objectPositionY + eps < state->labels[1].objectPositionY) {
//		game->makeAction(actions[1]); //right
//	}
//	else {
//		game->makeAction(actions[2]); // shoot
//	}
//}

void game_init() {
	game->setViZDoomPath("../vizdoom/vizdoom");
	game->setDoomGamePath("../vizdoom/freedoom2.wad");
	game->loadConfig("../vizdoom/scenarios/task2.cfg"); // add configurations for game
	game->setScreenResolution(RES_640X480); // разрешение
	game->setLabelsBufferEnabled(1); // add this
	game->setWindowVisible(0); // exception with Linux without X Series
	game->setRenderWeapon(1); // is the gun will be in the game
	game->setRenderHud(1);
	game->init();
}

int find(cv::Mat screen) {
	CvPoint sum = cvPoint(0, 0);
	for (int x = 200; x < (&screen)->cols; ++x)
		for (int y = 200; y < 250; ++y)
			if (screen.at<unsigned char>(y, x) == 255) {
				sum.x += x + 15;
				return sum.x;
			}
}

void kill(int x) {
	if (x > 365)
		game->makeAction(actions[1]); //right
	else if (x < 275)
		game->makeAction(actions[0]); //left
	else
		game->makeAction(actions[2]); //shoot
}

void RunTask1(int episodes)
{
	game_init();

	auto image = cv::Mat(480, 640, CV_8UC3);
	auto greyscale = cv::Mat(480, 640, CV_8UC1);

	for (auto i = 0; i < episodes; i++)
	{
		game->newEpisode();
		std::cout << "Episode #" << i + 1 << std::endl;
		while (!game->isEpisodeFinished())
		{
			const auto& gamestate = game->getState();
			std::memcpy(image.data, gamestate->screenBuffer->data(), gamestate->screenBuffer->size());

			cv::extractChannel(image, greyscale, 1);
			cv:threshold(greyscale, greyscale, 130, 255, cv::THRESH_BINARY);


			cv::moveWindow("Game", 60, 20);
			cv::moveWindow("Greyscale", 710, 20);
			imshow("Game", image);
			imshow("Greyscale", greyscale);
			int x = find(greyscale);
			kill(x);
			//find_demon_and_kill(gamestate);


			/*float y1 = centers[0].y;
			float y2 = centers[1].y;
			double eps = 21;
			if (y2 < y1)
			{

				if (centers[0].x + 36 < centers[1].x) {
					game->makeAction(actions[1]);
				}
				else if (centers[0].x - 30 > centers[1].x) {
					game->makeAction(actions[0]);
				}
				else {
					game->makeAction(actions[2]);
				}
			}
			else {

				if (centers[1].x + 35 < centers[0].x) {
					game->makeAction(actions[1]);
				}
				else if (centers[1].x - 30 > centers[0].x) {
					game->makeAction(actions[0]);
				}
				else {
					game->makeAction(actions[2]);
				}
			}*/

			char c = cv::waitKey(sleepTime);
			if (c == 27) break;
		}
		std::cout << game->getTotalReward() << std::endl;
		total_reward += game->getTotalReward();
	}
	std::cout << total_reward / episodes << endl;
	cvWaitKey(45);
}

void RunTask2(int episode)
{
	try
	{
		game->loadConfig(path + "\\scenarios\\task2.cfg");
		game->setWindowVisible(true);
		game->setRenderWeapon(true);
		game->init();
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}

	std::vector<double> actions = { 0,0,0,0 };

	double integral = 0;

	auto image = cv::Mat(480, 640, CV_8UC3);
	auto greyscale = cv::Mat(480, 640, CV_8UC1);

	cv::Mat clusters;

	for (auto i = 0; i < episode; i++)
	{
		game->newEpisode();
		std::cout << "Episode #" << i + 1 << std::endl;

		while (!game->isEpisodeFinished())
		{
			const auto& gameState = game->getState();
			std::memcpy(image.data, gameState->screenBuffer->data(), gameState->screenBuffer->size());

			std::vector<cv::Point2f> centers;
			std::vector<cv::Point2f> points(0);
			for (int x = 0; x < 640; x++)
			{
				for (int y = 0; y < 480; y++)
				{
					if (int(image.at<cv::Vec3b>(y, x)[2]) > 130 && int(image.at<cv::Vec3b>(y, x)[0]) < 50)
					{
						greyscale.at<unsigned char>(y, x) = 255;
						points.push_back(cv::Point2f(x, y));
					}
					else
					{
						greyscale.at<unsigned char>(y, x) = 0;
					}
				}
			}

			greyscale.convertTo(greyscale, CV_32F);

			cv::Mat samples = greyscale.reshape(1, greyscale.total());

			cv::kmeans(points, 1, clusters, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0), 3, cv::KMEANS_RANDOM_CENTERS, centers);

			greyscale.convertTo(greyscale, CV_8UC3);

			for (int i = 0; i < centers.size(); i++)
			{
				cv::Point c = centers[i];

				cv::circle(image, c, 5, cv::Scalar(0, 0, 255), -1, 8);
				cv::rectangle(image, cv::Rect(c.x - 25, c.y - 25, 50, 50), cv::Scalar(0, 0, 255));
			}

			for (int i = 0; i < points.size(); i++)
			{
				cv::circle(image, points[i], 2, cv::Scalar(0, 255, 0));
			}


			imshow("Game", image);
			imshow("Greyscale", greyscale);
			cv::moveWindow("Game", 60, 20);
			cv::moveWindow("Greyscale", 710, 20);

			double err = centers[0].x - 320;
			double p = err * 0.2;
			integral = integral + err * 0.01;
			double u = p + integral;
			actions = { 0, 0, u, 0 };
			if (abs(centers[0].x - 320) < 40)
			{
				actions = { 0,0,0,1 };
			}

			game->makeAction(actions);

			cv::waitKey(sleepTime);
		}
		std::cout << game->getTotalReward() << std::endl;
		total_reward += game->getTotalReward();
	}
}

int main()
{
	game->setViZDoomPath(path + "\\vizdoom.exe");
	game->setDoomGamePath(path + "\\freedoom2.wad");

	cv::namedWindow("Output Window", cv::WINDOW_AUTOSIZE);
	
	auto episodes = 10;
	RunTask2(episodes);
	
	game->close();

}