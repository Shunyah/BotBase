#include <iostream>
#include <opencv2/opencv.hpp>
#include <ViZDoom.h>
#include <chrono>
#include <thread>

vizdoom::DoomGame* game = new vizdoom::DoomGame();
//auto game = std::make_unique<vizdoom::DoomGame>();
auto screenBuff = cv::Mat(480, 640, CV_8UC3);
const unsigned int sleepTime = 1000 / vizdoom::DEFAULT_TICRATE;

/*void game_init() {
	game->setScreenResolution(vizdoom::RES_640X480);
	game->setLabelsBufferEnabled(true); // add this
	game->setWindowVisible(true);
	game->setRenderWeapon(true);
	game->setRenderHud(true);
}*/

void RunTask1(int episodes)
{
	try
	{
		game->loadConfig("../vizdoom/scenarios/task1.cfg");
		game->init();
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}

	auto greyscale = cv::Mat(480, 640, CV_8UC1);

	std::vector<double> action;

		for (auto i = 0; i < episodes; i++)
	{
		game->newEpisode();
		std::cout << "Episode #" << i + 1 << std::endl;

while (!game->isEpisodeFinished())
{
	const auto& gamestate = game->getState();

	std::memcpy(screenBuff.data, gamestate->screenBuffer->data(), gamestate->screenBuffer->size());

	cv::extractChannel(screenBuff, greyscale, 2);

cv:threshold(greyscale, greyscale, 175, 255, cv::THRESH_BINARY);

	cv::imshow("Output Window", greyscale);

	double reward = game->makeAction({ 0, 0, 1 });
	//double reward = game->makeAction({ 1 });

	std::cout << reward << " ";
	cv::waitKey(sleepTime);
};

		std::cout << std::endl << game->getTotalReward() << std::endl;
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
	delete game;
}
